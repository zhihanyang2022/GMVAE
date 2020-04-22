import json

def save_history_json(history_loss, num_components):
    with open(f'../smba_gmprior_effect_on_rec/rec_history_{num_components}.json', 'w+') as json_f:
        json.dump(history_loss, json_f)