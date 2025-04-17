for var in $(seq 1 100); do
    echo $var; 
    k6 run dog-registration/test/performance_test_ml.js; 
    bash ./dog-registration/test/append_test_result.sh; 
    sleep 3;
done    