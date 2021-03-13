import json


def _get_params(json_file):
    """
    Get parameters from JSON file. Commented lines in the JSON file that begin
    with // are ignored. Parameters are returned as a dictionary.
    """
    json_str = ''

    with open(json_file) as jfile:
        for line in jfile:
            if '//' not in line:
                json_str += line

    json_dict = json.loads(json_str)
    return json_dict


def main():
    """
    Run the 1D bubbling fluidized bed (BFB) gasification model.
    """
    params = _get_params('ss-bfbgasf/params.json')
    print('D =', params['D'])


if __name__ == '__main__':
    main()
