#/bin/bash

echo '[BUILDING]'

dir_pth=$(pwd)
for folder in experiments data .config;
do
    if [ !-d  $folder];
    then
        echo "${$folder^^}_HOME=$dir_pth/$folder"
        mkdir $dir_pth/$folder
        echo "${$folder^^}_HOME=$dir_pth/$folder" >> .env
    fi
done

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
