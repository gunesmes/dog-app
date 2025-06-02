for var in $(seq 1 100); do
    echo $var; 
    k6 run performance_test_ml.js; 
done    