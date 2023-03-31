for num in {1..80}
do
    python label_tool.py --continuous --scene-num $num
    sleep 1m
done
