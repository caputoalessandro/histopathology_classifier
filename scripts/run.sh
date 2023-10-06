C_NAME="${2:-caputo}"
GPU="${3:-2}"
I_NAME=caputo

echo "############################################# REMOVE OLD IMAGE ################################################"
docker container stop $C_NAME
docker container rm $C_NAME

if [ -z $1 ]
then
  echo "############################################## RUN SHELL #################################################"
  docker run -it  \
        --name $C_NAME \
        -v $HOME/histopatho-cancer-grading/docker_output/assets/:/assets \
        -v $HOME/data:/data \
        -v $HOME/histopatho-cancer-grading/config:/config \
        -v $HOME/histopatho-cancer-grading/src:/src \
        -v $HOME/histopatho-cancer-grading/tsv:/tsv \
        -v $HOME/data/models:/models \
        --entrypoint sh \
        eidos-service.di.unito.it/caputo/$I_NAME

elif [ $GPU == '0' ] || [ $GPU == '1' ]
then
    echo "############################################## RUN ON GPU $GPU #################################################"
    docker run -it  \
    --ipc="host"  \
	--name $C_NAME \
	-e "PYTHONUNBUFFERED=1" \
	--gpus device=$GPU \
	-v $HOME/histopatho-cancer-grading/docker_output/assets:/assets \
	-v $HOME/histopatho-cancer-grading/config:/config \
	-v $HOME/data:/data \
	-v $HOME/histopatho-cancer-grading/src:/src \
	-v $HOME/histopatho-cancer-grading/tsv:/tsv \
    -v $HOME/data/models:/models \
    eidos-service.di.unito.it/caputo/$I_NAME /src/$1 ${@:4}

else
  echo "############################################## RUN WITHOUT GPU #################################################"
  docker run -it  \
    --ipc="host"  \
    --name $C_NAME \
    -e "PYTHONUNBUFFERED=1" \
    -v $HOME/histopatho-cancer-grading/docker_output/assets:/assets \
    -v $HOME/histopatho-cancer-grading/config:/config \
    -v $HOME/data:/data \
    -v $HOME/histopatho-cancer-grading/src:/src \
    -v $HOME/histopatho-cancer-grading/tsv:/tsv \
    -v $HOME/data/models:/models \
    eidos-service.di.unito.it/caputo/$I_NAME /src/$1 ${@:3}

fi
echo "############################################## FINISH #################################################"
