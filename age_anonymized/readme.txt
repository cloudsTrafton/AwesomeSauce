Keystroke dynamics dataset labeled by age groups
Author: Avar Pentel

Keystroke data is collected through different real life online systems between 2011 and 2018. Data logs in this set are grouped to 6 different folder representing 6 age groups -15, 16-19, 20-29, 30-39, 40-49, and 50+. There are 2.3 million keystrokes from 7119 keystroke data logs, produced by ca 1000 individual subjects.

Each log is presented as CSV file, where each row represents one key press. Each row consist of 3 numbers: first is the key code, second is the time in milliseconds the key was hold down (key up - key down), and the third is the time between the last key release to this key press.

Dataset: keystroke logs categorized by 6 age groups (different folders)

Please refer to paper: A. Pentel. 2018. Predicting User Age by Keystroke Dynamics. Advances in Intelligent Systems and Computing. Springer (in press)

Part of this dataset was also used in paper: A. Pentel. 2017. Predicting Age and Gender by Keystroke Dynamics and Mouse Patterns. In proceedings of UMAP '17 of the 25th Conference on User Modeling, Adaptation and Personalization. Pages 381-385. ACM Digital Library

Included preprocessing script keystroke_feature_extractor.js is created for MS Windows only and is supposed to run in console over cscript.exe, run_script.bat will start it right. Running directly by clicking on keystroke_feature_extractor.js file is possible, but as it is now, it will pop up new dialog for each log. To prevent this, all WScript.Echo lines in script need to be commented out, except last one, that informs, when all logs are processed.

There is also need to modify the script according to right set of features (array features), and to choose which age groups to process (array classes). 
This script calculates average seek, hold and n-graph latencies time for each key or sequence of keys. In feature array these have to be presented as single keycode number (65 for a, for instance), or multiple numbers separated with underline (65_66 for ab or 65_66_67 for abc). For each single number feature the average seek and hold time will be calculated, for multiple number features n-graph latency will be calculated.

Processing takes some time.

After finished, the script creates .arff file, which is suitable for Weka machine learning package. However, some more preprocessing might be needed, for example, replacing missing values. 

Feel free to modify this script to fit your purpose.
