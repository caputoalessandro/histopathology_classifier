I_NAME=caputo

echo "############################################# REMOVE OLD CONTAINER ################################################"
docker rmi eidos-service.di.unito.it/caputo/$I_NAME

echo "############################################# BUILD ################################################"
DOCKER_BUILDKIT=1 docker build $HOME/histopatho-cancer-grading -f $HOME/histopatho-cancer-grading/scripts/Dockerfile \
 -t eidos-service.di.unito.it/caputo/$I_NAME

#echo "############################################## PUSH #################################################"
#docker push eidos-service.di.unito.it/caputo/$I_NAME