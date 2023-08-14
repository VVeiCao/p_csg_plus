import os

routes = [{}, {}, {}, {}, {}, {}, {}, {}]
routes[0]["training_routes/routes_town01_long.xml"] = "scenarios/town01_all_scenarios.json"
routes[1]["training_routes/routes_town02_long.xml"] = "scenarios/town02_all_scenarios.json"
routes[2]["training_routes/routes_town03_long.xml"] = "scenarios/town03_all_scenarios.json"
routes[3]["training_routes/routes_town04_long.xml"] = "scenarios/town04_all_scenarios.json"
routes[4]["training_routes/routes_town05_long.xml"] = "scenarios/town05_all_scenarios.json"
routes[5]["training_routes/routes_town06_long.xml"] = "scenarios/town06_all_scenarios.json"
routes[6]["training_routes/routes_town07_long.xml"] = "scenarios/town07_all_scenarios.json"
routes[7]["training_routes/routes_town10_long.xml"] = "scenarios/town10_all_scenarios.json"


routes_list = [[], [], [], [], [], [], [], []]
for i in range(8):
    for route in routes[i]:
        routes_list[i].append(route.split("/")[1].split(".")[0])





if not os.path.exists("batch_run_long"):
    os.mkdir("batch_run_long")
for i in range(8):
    fw = open(f"batch_run_long/run_route{i}.sh", "w")
    for route in routes_list[i]:
        fw.write("bash data_collection/bashs/weather-mixed/%s.sh \n" % (route))

