import { check, sleep } from 'k6';
import http from 'k6/http';


export let options = {
    vus: 250, // number of virtual users
    duration: '30s', // duration of the test
    thresholds: {
        http_req_duration: [
            'avg<2', // average response time must be below 25ms
            'p(90)<3', // 90% of requests must complete below 35ms
            'p(95)<4', // 95% of requests must complete below 50ms
            'max<5' // max response time must be below 50ms
        ], 
        http_req_failed: [
            'rate<0.01' // http request failures should be less than 1%
        ], 
        checks: [
            'rate>0.99' // 99% of checks should pass
        ], 
    },
};

export default function () {
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

    console.log(`Response body: ${res.body}`); // Log the response body for debugging

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

    sleep(1);
}
