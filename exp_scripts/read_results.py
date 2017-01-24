import json
import re
def read_state(prefix):
    with open(prefix + '/latest_state.json') as ifp:
        state = json.load(ifp)
    train_ppl = []
    dm_ppl = []
    with open(prefix + '/training.log') as ifp:
        for line in ifp:
            if "Train ppl = " in line:
                m = re.search(r'(Train ppl = )([^,]*)', line)
                if m is not None:
                    train_ppl.append(m.group(2))
                else:
                    train_ppl.append("0.0")
                m = re.search(r'(DM PPL = )([^,]*)', line)
                if m is not None:
                    dm_ppl.append(m.group(2))
                else:
                    dm_ppl.append("0.0")
    print('{}\t{}\t{}\t \t{} ({})\t{}'.format(
        dm_ppl[state['best_epoch']], train_ppl[state['best_epoch']], 
        state['best_val_ppl'], state['best_epoch'] + 1,
        state['epoch'] + 1, prefix))
