for var in $(seq 1 100); do
    echo $var; 
    k6 run performance_test_data.js; 
    bash append_test_result.sh; 
    sleep 5;
done    