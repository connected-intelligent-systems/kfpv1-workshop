from os import environ
from datetime import datetime

import kfp
from deploykf import DeployKFCredentialsOutOfBand

class kfphelpers():
    def __init__(self, namespace: str, pipeline_yaml_path: str = 'default_pipeline.yaml', pl_name: str = 'default-pl-name') -> None:
       
        # Cluster access
        issuer = "https://10-101-20-33.sslip.io:443/dex"
        host = "https://10-101-20-33.sslip.io:443/pipeline"
        # initialize a credentials instance 
        credentials = DeployKFCredentialsOutOfBand(
            issuer_url=issuer, 
            skip_tls_verify=True,
        )

        # initialize a client instance for kfpv1
        self.client = DeployKFCredentialsOutOfBand.patched_kfp_client(verify_ssl=not credentials.skip_tls_verify)(
            host=host,
            credentials=credentials,
            namespace=namespace,
        )
        # Client/Pipeline
        self.pl_yaml_path = pipeline_yaml_path
        self.pl_name = pl_name

    '''
    [Deprecated] Creates 'quick' pipeline run without uploading or creating experiment (uses default)
    '''
    def create_quick_run(self, pipeline_function, run_params = dict()):
        kfp.compiler.Compiler().compile(pipeline_function, package_path=self.pl_yaml_path)
        self.client.create_run_from_pipeline_package(self.pl_yaml_path, arguments=run_params)

    '''
    Compiles pipieline from function --> Checks if pipeline exists
        / if False --> Creates pipeline with pl_name and pl_yaml_path
        / if True  --> Creates new pipeline version under the same pl_name
    '''
    def upload_pipeline(self, pipeline_function):
        kfp.compiler.Compiler().compile(pipeline_function, package_path=self.pl_yaml_path)
        # check if pipeline already exists in the cluster
        existing_pipeline_id = self.client.get_pipeline_id(self.pl_name)

        # upload pipeline to cluster
        if existing_pipeline_id is None:
            api_pl = self.client.upload_pipeline(pipeline_package_path=self.pl_yaml_path, pipeline_name=self.pl_name)
            print(f'Uploaded pipeline received id {api_pl.pipeline_id}')

        # Automatically create and upload new version of pipeline.
        else:
            # Version name is based on number of existing pipelines.
            versions_request_to_dict = self.client.list_pipeline_versions(existing_pipeline_id).to_dict()
            version = '_version_' + str(versions_request_to_dict['total_size'])

            # upload pipeline version
            api_pl = self.client.upload_pipeline_version(pipeline_package_path=self.pl_yaml_path, pipeline_name=self.pl_name, pipeline_version_name=self.pl_name+version)
            print(f'Uploaded pipeline received id {api_pl.pipeline_id}')
            print('Pipeline already exists. Automatically created a new version.')
            
    '''
    Create new pipeline run under the given *experiment_name*. Creates new expiriment if *experiment_name* doesnt exist.
    ''' 
    def create_run(self, pipeline_function, experiment_name='default-expirement', run_params = dict ()):
        # compile pipeline function
        kfp.compiler.Compiler().compile(pipeline_function, package_path=self.pl_yaml_path)
        # get/create experiment
        exp_id = self.create_experiment(experiment_name=experiment_name)
        # creating run
        api_run = self.client.run_pipeline(
            pipeline_package_path=self.pl_yaml_path,
            experiment_id=exp_id,
            params=run_params,
            job_name=f'{self.pl_name} {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
        )
        print(f'Started new run with id {api_run.id}.')

    '''
    Creates/fetches experiment
    '''
    def create_experiment(self, experiment_name):
        # creating or fetching an experiment
        exp_id = self.get_exp_id(experiment_name)
        if exp_id is None:
            api_exp = self.client.create_experiment(name=experiment_name)
            print(dir(exp_id))
            exp_id = api_exp.experiment_id
            print(f'Created Experiment {experiment_name} with id {api_exp.experiment_id}.')
            return api_exp.experiment_id
        print(f'Experiment {experiment_name} already exists with id {exp_id}')
        return exp_id

    '''
    Searches for experiment (*exp_name*) by iterating and return id
    '''
    def get_exp_id(self, exp_name):
        exps = self.client.list_experiments().experiments
        if exps == None: 
            return None
        else:
            for exp in exps:
                if exp.name == exp_name:
                    return exp.id
            return None
        
    '''
    Combination of *upload_pipeline()* and *create_run()*
    '''
    def upload_and_run(self, pipeline_function, experiment_name='default-expirement', run_params = dict ()):
        self.upload_pipeline(pipeline_function=pipeline_function)
        self.create_run(pipeline_function, experiment_name, run_params)
        