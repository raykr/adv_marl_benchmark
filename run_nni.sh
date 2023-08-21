# ps -A | grep Main_Thread | awk '{print $1}' | xargs kill -9
nnictl create --config nni-explore.yaml --port 8081