#/bin/bash

echo '[TESTING]';

poetry run pytest --no-header -v

if [ $? = 0 ] 
then
    echo '[TESTING] OK';
    echo 'commiting'
    # pegar depois como parametro
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

# export POETRY_DOTENV_LOCATION=.env && poetry run python experiments/adult.py

echo $(cat $POETRY_DOTENV_LOCATION)