# ps -A | grep Main_Thread | awk '{print $1}' | xargs kill -9
# nnictl create --config ./nni/configs/nni-explore-mappo.yaml --port 8080
# nnictl create --config ./nni/configs/nni-optimizer-gamma.yaml --port 8087
# nnictl create --config ./nni/configs/nni-optimizer-lr.yaml --port 8088

# traitor
nnictl create --config ./nni/configs/traitor/nni-explore-mappo.yaml --port 8090
nnictl create --config ./nni/configs/traitor/nni-optimizer-gamma.yaml --port 8091
nnictl create --config ./nni/configs/traitor/nni-optimizer-lr.yaml --port 8092