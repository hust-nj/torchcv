# 复制新文件，更改文件名并替换特定文件内容
for file in `ls mbv2_*.sh`
do
    newfile=`echo $file | sed 's/nonlocalnowd/basenet/g'`
    cp $file $newfile
    sed -i 's/nonlocalnowd/basenet/g' $newfile
done