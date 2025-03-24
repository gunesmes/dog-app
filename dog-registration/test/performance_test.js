import { check, sleep } from 'k6';
import http from 'k6/http';

export let options = {
    vus: 10, // number of virtual users
    duration: '10s', // duration of the test
    thresholds: {
        http_req_duration: [
            'avg<200', // average response time must be below 2ms
            'p(90)<300', // 90% of requests must complete below 3ms
            'p(95)<400', // 95% of requests must complete below 4ms
            'max<500' // max response time must be below 5ms
        ], 
        http_req_failed: [
            'rate<0.01' // http request failures should be less than 1%
        ], 
        checks: [
            'rate>0.99' // 99% of checks should pass
        ], 
    },
};

function registerDog() {
    let url = 'http://localhost:8084/register';
    let random_id = Math.floor(Math.random() * 1000);    
    let json_data = {
        name: `Dog-${random_id}`,
        breed: `Breed-${random_id}`,
        age: Math.floor(Math.random() * 15) + 1,
    };

    let payload = JSON.stringify(json_data);
    let params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    let res = http.post(url, payload, params);

    check(res, {
        'is status 201': (r) => r.status === 201,
        'is registered': (r) => {
            try {
                let responseBody = JSON.parse(r.body);
                return responseBody.id !== undefined
                    && responseBody.name === json_data.name
                    && responseBody.breed === json_data.breed
                    && responseBody.age === json_data.age;
            } catch (e) {
                console.error(`Failed to parse response body: ${e}`);
                return false;
            }
        },
    });

    return res.json().id;
}

function getRegisteredDogs() {
    let url = 'http://localhost:8084/dogs';
    let res = http.get(url);

    check(res, {
        'is status 200': (r) => r.status === 200,
        'has dogs': (r) => {
            try {
                let responseBody = JSON.parse(r.body);
                return Array.isArray(responseBody) && responseBody.length > 0;
            } catch (e) {
                console.error(`Failed to parse response body: ${e}`);
                return false;
            }
        },
    });
}

function getDogById(dogId) {
    let url = `http://localhost:8084/dogs/${dogId}`;
    let res = http.get(url);

    check(res, {
        'is status 200': (r) => r.status === 200,
        'is correct dog': (r) => {
            try {
                let responseBody = JSON.parse(r.body);
                return responseBody.id === dogId;
            } catch (e) {
                console.error(`Failed to parse response body: ${e}`);
                return false;
            }
        },
    });
}

export default function () {
    let dogId = registerDog();
    sleep(1);
    getRegisteredDogs();
    sleep(1);
    getDogById(dogId);
    sleep(1);
}
