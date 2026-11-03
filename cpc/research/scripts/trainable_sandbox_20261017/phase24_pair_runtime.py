"""Additional no-read guard for the independently selected paired chemical models."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).parent))
from phase24_property_runtime import worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=2,required=True);a=p.parse_args();blocked=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)) and '/paired_validation/' in str(args[0]):
            blocked.append(str(args[0]));raise RuntimeError('Paired chemical training files forbidden at prediction')
    sys.addaudithook(guard);worker(*a.worker)
    output=Path(a.worker[1]);value=json.loads(output.read_text());value['paired_training_blocked_paths']=blocked;output.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':main()
