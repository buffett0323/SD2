# Results Comparison

## Dataset: jsb_hard

| Metric                    | LAVE   |
| ------------------------- | ------ |
| Benchmark total           | 10     |
| Skipped (grammar-invalid) | 0      |
| Evaluated (n)             | 10     |
| Valid (count)             | 3      |
| Validity (%)              | 30.0%  |
| Timeouts (>120s)          | 1      |
| Mean resamples            | 18.60  |
| Mean time (s)             | 29.11  |
| Median time (s)           | 0.88   |
| P95 time (s)              | 93.59  |
| Max time (s)              | 120.00 |

### Per-instance Validity Agreement

Each row shows a validity pattern across methods (LAVE).

| Pattern (LAVE) | Count | %     | Example IDs             |
| -------------- | ----- | ----- | ----------------------- |
| ✗              | 7     | 70.0% | o10293, o10296, o10346… |
| ✓              | 3     | 30.0% | o10499, o1051, o1052    |

## Dataset: jsb_medium

| Metric                    | LAVE   | Dgrammar | DPGrammar |
| ------------------------- | ------ | -------- | --------- |
| Benchmark total           | 586    | 586      | 586       |
| Skipped (grammar-invalid) | 75     | 75       | 75        |
| Evaluated (n)             | 511    | 511      | 511       |
| Valid (count)             | 397    | 434      | 487       |
| Validity (%)              | 77.7%  | 84.9%    | 95.3%     |
| Timeouts (>120s)          | 68     | 0        | 0         |
| Mean resamples            | 233.87 | 32.43    | 2.11      |
| Mean time (s)             | 41.12  | 13.38    | 16.14     |
| Median time (s)           | 27.34  | 13.65    | 15.61     |
| P95 time (s)              | 120.00 | 20.93    | 29.36     |
| Max time (s)              | 120.05 | 33.45    | 88.17     |

### Per-instance Validity Agreement

Each row shows a validity pattern across methods (LAVE, Dgrammar, DPGrammar).

| Pattern (LAVE / Dgrammar / DPGrammar) | Count | %     | Example IDs             |
| ------------------------------------- | ----- | ----- | ----------------------- |
| ✓  ✓  ✓                               | 347   | 67.9% | o10217, o10518, o10617… |
| ✗  ✓  ✓                               | 70    | 13.7% | o10297, o11795, o12286… |
| ✓  ✗  ✓                               | 38    | 7.4%  | o1050, o12241, o13189…  |
| ✗  ✗  ✓                               | 32    | 6.3%  | o10927, o11689, o11975… |
| ✓  ✓  ✗                               | 11    | 2.2%  | o24180, o26197, o30006… |
| ✗  ✗  ✗                               | 6     | 1.2%  | o27216, o27789, o33011… |
| ✗  ✓  ✗                               | 6     | 1.2%  | o33928, o33932, o40220… |
| ✓  ✗  ✗                               | 1     | 0.2%  | o30439                  |

### Disagreement Cases

| Instance ID | LAVE | Dgrammar | DPGrammar |
| ----------- | ---- | -------- | --------- |
| o10297      | ✗    | ✓        | ✓         |
| o1050       | ✓    | ✗        | ✓         |
| o10927      | ✗    | ✗        | ✓         |
| o11689      | ✗    | ✗        | ✓         |
| o11795      | ✗    | ✓        | ✓         |
| o11975      | ✗    | ✗        | ✓         |
| o12241      | ✓    | ✗        | ✓         |
| o12286      | ✗    | ✓        | ✓         |
| o12496      | ✗    | ✓        | ✓         |
| o12610      | ✗    | ✓        | ✓         |
| o13         | ✗    | ✓        | ✓         |
| o13189      | ✓    | ✗        | ✓         |
| o13402      | ✗    | ✓        | ✓         |
| o14474      | ✓    | ✗        | ✓         |
| o14478      | ✗    | ✓        | ✓         |
| o15131      | ✗    | ✓        | ✓         |
| o17539      | ✗    | ✗        | ✓         |
| o20375      | ✗    | ✗        | ✓         |
| o20460      | ✗    | ✓        | ✓         |
| o21062      | ✗    | ✓        | ✓         |
| o21095      | ✗    | ✗        | ✓         |
| o21100      | ✗    | ✓        | ✓         |
| o21225      | ✓    | ✗        | ✓         |
| o21285      | ✗    | ✓        | ✓         |
| o21846      | ✓    | ✗        | ✓         |
| o24180      | ✓    | ✓        | ✗         |
| o26197      | ✓    | ✓        | ✗         |
| o26594      | ✓    | ✗        | ✓         |
| o27362      | ✗    | ✗        | ✓         |
| o29812      | ✗    | ✓        | ✓         |
| o30006      | ✓    | ✓        | ✗         |
| o30180      | ✓    | ✓        | ✗         |
| o30439      | ✓    | ✗        | ✗         |
| o30701      | ✓    | ✓        | ✗         |
| o30758      | ✗    | ✓        | ✓         |
| o30761      | ✗    | ✓        | ✓         |
| o30877      | ✗    | ✗        | ✓         |
| o31090      | ✗    | ✓        | ✓         |
| o31100      | ✗    | ✓        | ✓         |
| o31136      | ✗    | ✓        | ✓         |
| o31835      | ✗    | ✓        | ✓         |
| o33928      | ✗    | ✓        | ✗         |
| o33932      | ✗    | ✓        | ✗         |
| o34336      | ✓    | ✗        | ✓         |
| o36073      | ✗    | ✓        | ✓         |
| o36440      | ✗    | ✓        | ✓         |
| o37721      | ✗    | ✓        | ✓         |
| o3907       | ✓    | ✗        | ✓         |
| o39078      | ✗    | ✓        | ✓         |
| o39137      | ✓    | ✓        | ✗         |
| o393        | ✗    | ✓        | ✓         |
| o39500      | ✓    | ✗        | ✓         |
| o39780      | ✓    | ✓        | ✗         |
| o40220      | ✗    | ✓        | ✗         |
| o41780      | ✗    | ✗        | ✓         |
| o41800      | ✗    | ✓        | ✓         |
| o4264       | ✓    | ✗        | ✓         |
| o42985      | ✗    | ✗        | ✓         |
| o43729      | ✗    | ✓        | ✓         |
| o44206      | ✗    | ✓        | ✓         |
| o44462      | ✗    | ✓        | ✗         |
| o44947      | ✓    | ✓        | ✗         |
| o45199      | ✗    | ✓        | ✓         |
| o45200      | ✗    | ✓        | ✓         |
| o45752      | ✗    | ✓        | ✗         |
| o45806      | ✓    | ✗        | ✓         |
| o47263      | ✗    | ✓        | ✓         |
| o48073      | ✗    | ✓        | ✓         |
| o48116      | ✗    | ✗        | ✓         |
| o48339      | ✓    | ✗        | ✓         |
| o4850       | ✓    | ✓        | ✗         |
| o49232      | ✓    | ✓        | ✗         |
| o49732      | ✗    | ✓        | ✓         |
| o5165       | ✗    | ✓        | ✓         |
| o5223       | ✓    | ✓        | ✗         |
| o52827      | ✗    | ✗        | ✓         |
| o5344       | ✗    | ✗        | ✓         |
| o5352       | ✓    | ✗        | ✓         |
| o5369       | ✓    | ✗        | ✓         |
| o53702      | ✓    | ✗        | ✓         |
| o5375       | ✗    | ✗        | ✓         |
| o5395       | ✗    | ✓        | ✓         |
| o54557      | ✗    | ✓        | ✓         |
| o54973      | ✗    | ✓        | ✓         |
| o55102      | ✗    | ✓        | ✓         |
| o55595      | ✗    | ✓        | ✓         |
| o56232      | ✗    | ✓        | ✓         |
| o57650      | ✗    | ✓        | ✓         |
| o58601      | ✗    | ✗        | ✓         |
| o60099      | ✗    | ✓        | ✓         |
| o60173      | ✗    | ✓        | ✓         |
| o60864      | ✓    | ✗        | ✓         |
| o60973      | ✓    | ✗        | ✓         |
| o60997      | ✗    | ✗        | ✓         |
| o61001      | ✗    | ✓        | ✓         |
| o61616      | ✗    | ✓        | ✓         |
| o61636      | ✗    | ✓        | ✓         |
| o61638      | ✗    | ✓        | ✓         |
| o6230       | ✗    | ✗        | ✓         |
| o6243       | ✓    | ✗        | ✓         |
| o6256       | ✓    | ✗        | ✓         |
| o63181      | ✗    | ✗        | ✓         |
| o64726      | ✗    | ✓        | ✓         |
| o65372      | ✗    | ✗        | ✓         |
| o65504      | ✓    | ✗        | ✓         |
| o65751      | ✗    | ✓        | ✓         |
| o66330      | ✗    | ✗        | ✓         |
| o66610      | ✗    | ✓        | ✓         |
| o67599      | ✗    | ✗        | ✓         |
| o68447      | ✗    | ✗        | ✓         |
| o69100      | ✗    | ✗        | ✓         |
| o69161      | ✗    | ✗        | ✓         |
| o69763      | ✓    | ✗        | ✓         |
| o69971      | ✗    | ✓        | ✓         |
| o69975      | ✗    | ✓        | ✓         |
| o69991      | ✗    | ✓        | ✓         |
| o70369      | ✗    | ✓        | ✓         |
| o70372      | ✗    | ✗        | ✓         |
| o70379      | ✗    | ✓        | ✓         |
| o72521      | ✗    | ✓        | ✓         |
| o73018      | ✓    | ✗        | ✓         |
| o7377       | ✗    | ✓        | ✓         |
| o7381       | ✓    | ✗        | ✓         |
| o73952      | ✓    | ✗        | ✓         |
| o73958      | ✗    | ✗        | ✓         |
| o74180      | ✓    | ✗        | ✓         |
| o74424      | ✗    | ✗        | ✓         |
| o7540       | ✓    | ✗        | ✓         |
| o75601      | ✗    | ✓        | ✗         |
| o7633       | ✗    | ✓        | ✓         |
| o76474      | ✓    | ✗        | ✓         |
| o78058      | ✓    | ✗        | ✓         |
| o78735      | ✗    | ✓        | ✓         |
| o78957      | ✓    | ✗        | ✓         |
| o79651      | ✗    | ✓        | ✓         |
| o80822      | ✓    | ✗        | ✓         |
| o81649      | ✗    | ✓        | ✓         |
| o82154      | ✓    | ✗        | ✓         |
| o82311      | ✗    | ✓        | ✓         |
| o82635      | ✗    | ✓        | ✓         |
| o82712      | ✗    | ✓        | ✓         |
| o83272      | ✗    | ✓        | ✓         |
| o83282      | ✓    | ✗        | ✓         |
| o84343      | ✗    | ✗        | ✓         |
| o87934      | ✓    | ✗        | ✓         |
| o88958      | ✓    | ✗        | ✓         |
| o90425      | ✗    | ✓        | ✓         |
| o90951      | ✗    | ✗        | ✓         |
| o9775       | ✗    | ✗        | ✓         |
| o9795       | ✓    | ✗        | ✓         |
| o9796       | ✓    | ✗        | ✓         |
| o9852       | ✗    | ✓        | ✓         |
| o9874       | ✗    | ✓        | ✓         |
| o9875       | ✗    | ✓        | ✓         |
| o9881       | ✗    | ✗        | ✓         |
| o9886       | ✗    | ✓        | ✓         |
| o9932       | ✗    | ✗        | ✓         |
| o9944       | ✓    | ✗        | ✓         |

## Dataset: jsonschema

| Metric                    | LAVE   |
| ------------------------- | ------ |
| Benchmark total           | 10     |
| Skipped (grammar-invalid) | 0      |
| Evaluated (n)             | 10     |
| Valid (count)             | 10     |
| Validity (%)              | 100.0% |
| Timeouts (>120s)          | 0      |
| Mean resamples            | 0.20   |
| Mean time (s)             | 7.17   |
| Median time (s)           | 6.46   |
| P95 time (s)              | 12.95  |
| Max time (s)              | 13.23  |

### Per-instance Validity Agreement

Each row shows a validity pattern across methods (LAVE).

| Pattern (LAVE) | Count | %      | Example IDs                                |
| -------------- | ----- | ------ | ------------------------------------------ |
| ✓              | 10    | 100.0% | jsonschema_0, jsonschema_1, jsonschema_10… |

## File Inventory

| Group (base name)                             | Method                                        | Dataset    | Shards | Records |
| --------------------------------------------- | --------------------------------------------- | ---------- | ------ | ------- |
| lave_timed_jsb_hard_s0_t128                   | lave                                          | jsb_hard   | 1      | 10      |
| lave_timed_jsb_medium_s0_t128                 | lave                                          | jsb_medium | 12     | 790     |
| lave_timed_jsonschema_s0_t128                 | lave                                          | jsonschema | 1      | 10      |
| v2_async_ac4_timed_jsb_medium_s0_t128         | dgrammar_v2_async                             | jsb_medium | 12     | 691     |
| dp_jsb_medium_s0_t128                         | dgrammar_dp                                   | jsb_medium | 12     | 691     |
| dp_jsb_medium_s1_t128                         | dgrammar_dp                                   | jsb_medium | 1      | 3       |
| exp_A_jsb_medium_s0_t128                      | exp_A_jsb_medium_s0_t128                      | jsb_medium | 1      | 16      |
| exp_B_jsb_medium_s0_t128                      | exp_B_jsb_medium_s0_t128                      | jsb_medium | 2      | 20      |
| lave_combined_timed_jsb_hard_s0_t128          | lave_combined                                 | jsb_hard   | 2      | 14      |
| lave_combined_timed_jsonschema_s0_t128        | lave_combined                                 | jsonschema | 1      | 10      |
| lave_dir1_timed_jsonschema_s0_t128            | lave_dir1                                     | jsonschema | 1      | 10      |
| lave_dir2_timed_jsonschema_s0_t128            | lave_dir2                                     | jsonschema | 1      | 10      |
| lave_dir3_timed_jsonschema_s0_t128            | lave_dir3                                     | jsonschema | 1      | 10      |
| lave_dir4_timed_jsonschema_s0_t128            | lave_dir4                                     | jsonschema | 1      | 10      |
| lave_fn_detection_jsb_hard_s0_t128_oml12      | lave_fn_detection_jsb_hard_s0_t128_oml12      | jsb_hard   | 1      | 10      |
| lave_fn_detection_jsonschema_s0_t128_oml12    | lave_fn_detection_jsonschema_s0_t128_oml12    | jsonschema | 1      | 10      |
| lave_oracle_validate_jsb_medium_s0_t128_oml12 | lave_oracle_validate_jsb_medium_s0_t128_oml12 | jsb_medium | 1      | 6       |
