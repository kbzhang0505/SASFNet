echo "Circul Test is starting..."
declare -i i=50000
for ((;i<=103000;i=i+50))
do
  res=`python test.py --whitch_epoch $i`
  echo res >> test_result.txt
done
