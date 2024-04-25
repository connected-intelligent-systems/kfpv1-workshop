import base64
import hashlib
import json
import logging
import os
import sys
import time
from typing import Optional
import kfp
import requests
import urllib3
from kubernetes.client import configuration
from requests_oauthlib import OAuth2Session

try:
    # for kubeflow pipelines v2
    from kfp.client.token_credentials_base import TokenCredentialsBase
except ImportError:
    # for kubeflow pipelines v1
    from kfp.auth import TokenCredentialsBase


class DeployKFCredentialsOutOfBand(TokenCredentialsBase):
    """
    A Kubeflow Pipelines credential provider which uses an "out-of-band" OIDC login flow.

    WARNING: intended for deployKF clusters only, unlikely to work with other Kubeflow clusters.

    Key features:
     - uses the OIDC client named 'kubeflow-pipelines-sdk', which is pre-configured in deployKF
     - stores tokens in the user's home directory '~/.config/kfp/dkf_credentials.json'
       (this file is indexed by issuer URL, so multiple clusters can be used concurrently)
     - attempts to use the "refresh_token" grant before prompting the user to login again
       (in deployKF, refresh tokens are valid if used at least once every 7 days, and not longer than 90 days in total)
    """

    def __init__(self, issuer_url: str, skip_tls_verify: bool = False):
        """
        Initialize a DeployKFTokenCredentials instance.

        :param issuer_url: the OIDC issuer URL (e.g. 'https://deploykf.example.com:8443/dex')
        :param skip_tls_verify: if True, skip TLS verification
        """
        # oidc configuration
        self.oidc_issuer_url = issuer_url
        self.oidc_client_id = "kubeflow-pipelines-sdk"
        self.oidc_redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        self.oidc_scope = ["openid", "email", "groups", "profile", "offline_access"]

        # other configuration
        self.http_timeout = 15
        self.local_credentials_path = os.path.join(
            os.path.expanduser("~"), ".config", "kfp", "dkf_credentials.json"
        )

        # setup logging
        self.log = logging.getLogger(__name__)
        self._setup_logging()

        # disable SSL verification, if requested
        self.skip_tls_verify = skip_tls_verify
        if self.skip_tls_verify:
            self.log.warning("TLS verification is disabled")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

        # discover the OIDC issuer configuration
        self._discover_oidc()

        # perform the initial login, if necessary
        self.get_token()

    def _setup_logging(self):
        self.log.propagate = False
        self.log.setLevel(logging.INFO)
        if not self.log.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.log.addHandler(handler)

    def _discover_oidc(self):
        """
        Discover the OIDC issuer configuration.
        https://openid.net/specs/openid-connect-discovery-1_0.html
        """
        oidc_discovery_url = f"{self.oidc_issuer_url}/.well-known/openid-configuration"
        self.log.info("Discovering OIDC configuration from: %s", oidc_discovery_url)
        response = requests.get(
            url=oidc_discovery_url,
            timeout=self.http_timeout,
            verify=not self.skip_tls_verify,
        )
        response.raise_for_status()
        oidc_issuer_config = response.json()
        self.oidc_issuer = oidc_issuer_config["issuer"]
        self.oidc_auth_endpoint = oidc_issuer_config["authorization_endpoint"]
        self.oidc_token_endpoint = oidc_issuer_config["token_endpoint"]

    def _read_credentials(self) -> dict:
        """
        Read credentials from the JSON file for the current issuer.
        """
        self.log.debug(
            "Checking for existing credentials in: %s", self.local_credentials_path
        )
        if os.path.exists(self.local_credentials_path):
            with open(self.local_credentials_path, "r") as file:
                data = json.load(file)
                return data.get(self.oidc_issuer, {})
        return {}

    def _write_credentials(self, token: str):
        """
        Write the provided token to the local credentials file (under the current issuer).
        """
        # Create the directory, if it doesn't exist
        credential_dir = os.path.dirname(self.local_credentials_path)
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir, exist_ok=True)

        # Read all existing credentials from the JSON file
        credentials_data = {}
        if os.path.exists(self.local_credentials_path):
            with open(self.local_credentials_path, "r") as f:
                data = json.load(f)

        # Update the credentials for the given issuer
        credentials_data[self.oidc_issuer] = token
        self.log.info("Writing credentials to: %s", self.local_credentials_path)
        with open(self.local_credentials_path, "w") as f:
            json.dump(credentials_data, f)

    def _generate_pkce_verifier(self) -> (str, str):
        """
        Generate a PKCE code verifier and its derived challenge.
        https://tools.ietf.org/html/rfc7636#section-4.1
        """
        # Generate a code_verifier of length between 43 and 128 characters
        code_verifier = base64.urlsafe_b64encode(os.urandom(96)).decode("utf-8")
        code_verifier = code_verifier.rstrip("=")
        code_verifier = code_verifier[:128]

        # Generate the code_challenge using the S256 method
        sha256_digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = (
            base64.urlsafe_b64encode(sha256_digest).decode("utf-8").rstrip("=")
        )

        return code_verifier, code_challenge

    def _refresh_token(self, oauth_session: OAuth2Session) -> Optional[dict]:
        """
        Attempt to refresh the provided token.
        https://requests-oauthlib.readthedocs.io/en/latest/oauth2_workflow.html#refreshing-tokens
        """
        if not oauth_session.token.get("refresh_token", None):
            return None

        self.log.warning("Attempting to refresh token...")
        try:
            new_token = oauth_session.refresh_token(
                self.oidc_token_endpoint,
                client_id=self.oidc_client_id,
                timeout=self.http_timeout,
                verify=not self.skip_tls_verify,
            )
            self.log.info("Successfully refreshed token!")
            self._write_credentials(new_token)
            return new_token
        except Exception as ex:
            self.log.error("Failed to refresh token!", exc_info=ex)

    def _login(self, oauth_session: OAuth2Session) -> dict:
        """
        Start a new "out-of-band" login flow.
        """
        self.log.info("Starting new 'out-of-band' login flow...")

        verifier, challenge = self._generate_pkce_verifier()
        authorization_url, state = oauth_session.authorization_url(
            self.oidc_auth_endpoint,
            code_challenge_method="S256",
            code_challenge=challenge,
        )

        # ensure everything is printed to the console before continuing
        sys.stderr.flush()
        time.sleep(0.5)

        # Get the authorization code from the user
        print(
            f"\nPlease open this URL in a browser to continue:\n > {authorization_url}\n",
            flush=True,
        )
        user_input = input("Enter the authorization code:\n > ")
        authorization_code = user_input.strip()

        # Exchange the authorization code for a token
        new_token = oauth_session.fetch_token(
            self.oidc_token_endpoint,
            code=authorization_code,
            code_verifier=verifier,
            include_client_id=True,
            state=state,
            timeout=self.http_timeout,
            verify=not self.skip_tls_verify,
        )
        self.log.info("Successfully fetched new token!")
        self._write_credentials(new_token)
        return new_token

    def get_token(self) -> str:
        """
        Get the current auth token.
        Will attempt to use "refresh_token" before prompting the user to login again.
        """
        # return the existing token, if it's valid for at least 5 minutes
        stored_token = self._read_credentials()
        if stored_token:
            expires_at = stored_token.get("expires_at", 0)
            expires_in = expires_at - time.time()
            if expires_in > 300:
                self.log.info(
                    "Using cached auth token (expires in %d seconds)", expires_in
                )
                return stored_token["id_token"]
            elif expires_in > 0:
                self.log.warning(
                    "Existing auth token expires in %d seconds",
                    expires_in,
                )
            else:
                self.log.warning("Existing auth token has expired!")

        oauth_session = OAuth2Session(
            self.oidc_client_id,
            redirect_uri=self.oidc_redirect_uri,
            scope=self.oidc_scope,
            token=stored_token,
        )

        # try to refresh the token, or start a new login flow
        new_token = self._refresh_token(oauth_session)
        if not new_token:
            new_token = self._login(oauth_session)

        return new_token["id_token"]

    def refresh_api_key_hook(self, config: configuration.Configuration):
        config.verify_ssl = not self.skip_tls_verify
        config.api_key["authorization"] = self.get_token()
       
    # creates a patched client that supports disabling SSL verification
    # required before kfp v2: https://github.com/kubeflow/pipelines/pull/7174
    def patched_kfp_client(verify_ssl=True):
        _original_load_config = kfp.Client._load_config

        def _patched_load_config(client_self, *args, **kwargs):
            config = _original_load_config(client_self, *args, **kwargs)
            config.verify_ssl = verify_ssl
            return config

        _patched_client = kfp.Client
        _patched_client._load_config = _patched_load_config

        return _patched_client 
        
       
       