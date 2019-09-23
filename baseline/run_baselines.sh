for f in gated/*.inp; do
    python3 run_setting.py "$f" >> results100max.txt
done