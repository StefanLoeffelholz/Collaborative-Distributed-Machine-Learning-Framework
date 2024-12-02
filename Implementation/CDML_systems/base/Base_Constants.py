period = {"phase": 0, "signal": 1, "signal key value": 2}
period_interim_results = {"signal": 0, "detail": 1}
interim_results = {"result": 0, "detail": 1}
possible_results = {"Federated":"model", "Split":"split", "Assisted": "statistic"}
base_amount_of_roles = {"configurator": "1", "coordinator": "1", "selector": "1", "updater": "1", "trainer": "10"}
initialization_amount_of_roles = {"configurator": "0", "coordinator": "0", "selector": "0", "updater": "0", "trainer": "0"}
agent_tag_to_role = {"Tra": "trainer", "Coo": "coordinator", "Upd": "updater", "Con": "configurator", "Sel": "selector"}
possible_tags = {1: "Initialization done", 2: "ML Task", 3 : "Trainer to be Updated", 4: "Updater to be sent results", 5 :"Request specs!", 6: "Apply", 7:"Specs", 8:"Decission", 
                 9:"Agent selection", 10 :"Ready", 11: "Interim Result", 12: "Next Task", 13: "Selector", 14: "TraUpd Pairings", 15: "Update"}