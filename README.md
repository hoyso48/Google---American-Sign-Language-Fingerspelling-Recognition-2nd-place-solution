See: https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434588

| | troughput (iterations/s) | latency (ms) | model size (Mb)| Public LB | Private LB |
| --- | --- | --- | --- | --- | --- |
| CTCGreedy | 20.14 | 49.66 | 11.41 | 0.815 | 0.807 |
| ATTGreedy | 10.16 | 98.42 | 11.77 | 0.816 | 0.808 |
| CTCATTJointGreedy | 5.26 | 190.22 | 12.49 | 0.820 | 0.812 |
| CTCATTJointGreedy -2xseed | 2.71 | 368.95 | 24.90 | 0.825 | 0.817 |
| CTCATTJointGreedy -3xseed | 1.75 | 570.77 | 37.30 | 0.825 | 0.819 |
