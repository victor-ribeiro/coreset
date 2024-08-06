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
