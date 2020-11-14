import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--data-path', type=str,
                    dest='data_path', help='Path to the CSV input data. First row headers. First column IDs',
                    default='/spell/GEO_features.csv')

parser.add_argument('--num-nodes', default=[2000, 1000, 500], dest='N_NODES', metavar='N', type=int, nargs='+', help='Number of nodes in each layer.')
parser.add_argument('--dropout', default=[0.1],dest='DROPOUT', type=int, nargs='+', help='Number of nodes in each layer.')
parser.add_argument('--batch', default=15, dest='BATCH_SIZE', type=int, help='Number of samples per batch.')
parser.add_argument('--epochs', default=150, dest='EPOCHS', type=int, help='Number of epochs.')
parser.add_argument('--test', default=0.2, dest="TEST_RATIO", type=float, help='Ratio of samples kept out for testing.')
parser.add_argument('--verbose', default=2, dest="VERBOSITY", type=int, choices=[0,1,2], help='Verbosity level: 0 None, 1 Info, 2 All')
parser.add_argument('--tolerance', default=5, dest="PATIENCE", type=int, help='Tolenrance to the rate of improvment between each batch. Low values terminate quicker.')
args = parser.parse_args()

import spell.client
client = spell.client.from_environment()

def fit():
    return 1+1

client_run = client.runs.new(command='fit',
                github_url='https://github.com/jgeofil/SDAE-biomarkers.git',
                github_ref='master',
                pip_packages=['pandas', 'numpy', 'scipy', 'sklearn'],
                attached_resources={'uploads/sdae/': 'data/'},
                envvars=None,
                framework='tensorflow',
                framework_version=2.2)
                      
client_run

client_run.wait_status(*client.runs.FINAL)
client_run.refresh()
if client_run.status == client.runs.COMPLETE:
    print("Run succeeded!")
else:
    print("Run failed!")