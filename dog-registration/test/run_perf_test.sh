for var in $(seq 1 5); do
    echo $var; 
    k6 run performance_test_ml.js; 
    bash append_test_result.sh; 
    sleep 3;
done    