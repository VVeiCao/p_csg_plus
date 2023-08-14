import os

routes = [{}, {}, {}, {}, {}, {}, {}, {}]
routes[0]["training_routes/routes_town01_short.xml"] = "scenarios/town01_all_scenarios.json"
routes[0]["training_routes/routes_town01_tiny.xml"] = "scenarios/town01_all_scenarios.json"
routes[0]["training_routes/routes_town01_long.xml"] = "scenarios/town01_all_scenarios.json"
routes[1]["training_routes/routes_town02_short.xml"] = "scenarios/town02_all_scenarios.json"
routes[1]["training_routes/routes_town02_tiny.xml"] = "scenarios/town02_all_scenarios.json"
routes[1]["training_routes/routes_town02_long.xml"] = "scenarios/town02_all_scenarios.json"
routes[2]["training_routes/routes_town03_short.xml"] = "scenarios/town03_all_scenarios.json"
routes[2]["training_routes/routes_town03_tiny.xml"] = "scenarios/town03_all_scenarios.json"
routes[2]["training_routes/routes_town03_long.xml"] = "scenarios/town03_all_scenarios.json"
routes[3]["training_routes/routes_town04_short.xml"] = "scenarios/town04_all_scenarios.json"
routes[3]["training_routes/routes_town04_tiny.xml"] = "scenarios/town04_all_scenarios.json"
routes[3]["training_routes/routes_town04_long.xml"] = "scenarios/town04_all_scenarios.json"
routes[4]["training_routes/routes_town05_short.xml"] = "scenarios/town05_all_scenarios.json"
routes[4]["training_routes/routes_town05_tiny.xml"] = "scenarios/town05_all_scenarios.json"
routes[4]["training_routes/routes_town05_long.xml"] = "scenarios/town05_all_scenarios.json"
routes[5]["training_routes/routes_town06_short.xml"] = "scenarios/town06_all_scenarios.json"
routes[5]["training_routes/routes_town06_tiny.xml"] = "scenarios/town06_all_scenarios.json"
routes[5]["training_routes/routes_town06_long.xml"] = "scenarios/town06_all_scenarios.json"
routes[6]["training_routes/routes_town07_short.xml"] = "scenarios/town07_all_scenarios.json"
routes[6]["training_routes/routes_town07_tiny.xml"] = "scenarios/town07_all_scenarios.json"
routes[6]["training_routes/routes_town07_long.xml"] = "scenarios/town07_all_scenarios.json"
routes[7]["training_routes/routes_town10_short.xml"] = "scenarios/town10_all_scenarios.json"
routes[7]["training_routes/routes_town10_tiny.xml"] = "scenarios/town10_all_scenarios.json"
routes[7]["training_routes/routes_town10_long.xml"] = "scenarios/town10_all_scenarios.json"

routes_list = [[], [], [], [], [], [], [], []]
for i in range(8):
    for route in routes[i]:
        routes_list[i].append(route.split("/")[1].split(".")[0])




weathers = ['weather-mixed', 'clear-weather']
if not os.path.exists("batch_run"):
    os.mkdir("batch_run")
for w in weathers:
    if not os.path.exists(os.path.join("batch_run" , w)):
        os.mkdir(os.path.join("batch_run" , w))
for w in weathers:
    for i in range(8):
        fw = open(f"batch_run/{w}/run_route{i}.sh", "w")
        for route in routes_list[i]:
            fw.write(f"bash data_collection/bashs/{w}/{route}.sh \n")

