# ps -A | grep Main_Thread | awk '{print $1}' | xargs kill -9
nnictl create --config ./nni/configs/nni-explore-mappo.yaml --port 8080