#/bin/bash

##########    CRAIG BASELINE    ##########


# echo '[TESTING]';

# poetry run pytest --no-header -v

if [ ! -d "coreset/craig" ];
then
    git clone https://github.com/baharanm/craig.git
fi

if [ $? = 0 ] 
then
    echo '[TESTING] OK';
    echo 'commiting'
    git add .
    git commit -m 'coreset prototype'
    git push
else
    echo '[ERRO]' $?.
fi

echo '[BUILDING]'

dir_pth=$(pwd)
for name in experiments data .config;
do
    up_name=$(echo "$name" | tr '[:lower:]' '[:upper:]')_HOME
    if [ ! -d  "$dir_pth/$name" ];
    then
        echo [$name]
        mkdir "$dir_pth/$name"
        echo "$up_name=$dir_pth/$name" >> .env
    fi
done

# if [ -f data.tar.gz  ];
# then
#     echo unpacking data
#     # tar -xvf data.tar.gz data
# else
#     tar -cvf data.tar.gz data
# fi

# for folder in $(ls experiments);
# do
#    echo '[RUNNING]' experiments/$folder
#    poetry run python experiments/$folder
#    echo DONE
# done

# poetry run python experiments/nursery

# scp -r data.tar.gz vicorr@fatnode.dexl.lncc.br:/home/vicorr/tese/coreset/data/data.tar.gz
# rm data.tar.gz

# export POETRY_DOTENV_LOCATION=.env && poetry run python experiments/adult.py

# echo $(cat $POETRY_DOTENV_LOCATION)
