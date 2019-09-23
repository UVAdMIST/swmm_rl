# Baseline explorations

## What I did here?

baseline.py
- tries 100 different valve settings
- records the setting with the least flood

run_setting.py
- runs the given valve setting on the argument .inp file

run_baselines.sh
- runs run_setting.py on every file in gated

anything else
- probably not important

## Results?

The best setting for 25yr24-hour_100yrmax_0.inp is: [1, 3] 674.6305836538966 - meaning valves: 0.1, 0.3 returned a total flood of 674.6305836538966

Note that the setting is not unique, same result could be achieved by a different couple of valve settings.

then those settings were run on all .inp files via run_baselines.sh (results.txt)

The greatest amount of flood happened in 100yr12-hour_100yrmax_0.inp with 1950.5970592825622.

baseline.py is run on 100yr12-hour_100yrmax_0.inp resulting in [0, 1] 1832.5766654304844

run_baselines.sh was run again using these new settings (results100max.txt)

If compared with results.txt these settings reduced flood on some instances (~100 more .inp files did not produce flood compared to the previous run) and **increased** on some. (Both on instances with less and more overall rainfall than the two .inp tested above)

This exploration could work as a "disproof" of concept on the question:
- does the best static setup for the highest rainfall perform well on the rest of the rainfall data?


### Notes
- One given thing is that StormwaterEnv calculates the flood properly. I would like to think it does.

- We could do a bit more precise testing. (e.g. more precision on settings.)


## Conclusion

Static valve setup is not the "best" solution for this problem. Hurray! machine learning is still needed. Now I plan on doing some dynamic valve setup baseline models. to see how well they perform.

For now, baseline.py is the best non machine learning baseline model we have.

### Running instructions
for windows
-  ¯\\\_(ツ)_/¯ 

for linux 
- just run the files. gated and inpfiles folders are suppose to have .inp files but I'm untracking them. I expect them to be in the master

for mac
- pyswmm won't run on mac until someone figures out _that_ gcc error.
- Use docker and see "for linux"