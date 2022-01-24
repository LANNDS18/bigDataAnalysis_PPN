import json
from pyahp import parse

with open('examples/simple_with_subcriteria.json') as json_model:
    # model can also be a python dictionary
    model = json.load(json_model)

ahp_model = parse(model)
priorities = ahp_model.get_priorities()
print(priorities)
