import { check, sleep } from 'k6';
import http from 'k6/http';

let uniqueIdCounter = 0;

export let options = {
    vus: 1, // number of virtual users
    duration: '10s', // duration of the test
    thresholds: {
        http_req_duration: ['p(95)<20'], // 95% of requests must complete below 20ms
        http_req_failed: ['rate<0.01'], // http request failures should be less than 1%
        checks: ['rate>0.99'], // 99% of checks should pass
    },
};

function generateUniqueId() {
    return ++uniqueIdCounter;
}

export default function () {
    let url = 'http://localhost:8084/register';
    let id = generateUniqueId();
    let json_data = {
        'ID': `${id}`,
        'Name': `Dog-${id}`,
        'Breed': `Breed-${id}`,
    };

    let payload = JSON.stringify(json_data);
    let params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    let res = http.post(url, payload, params);
    check(res, {
        'is status 201': (res) => res.status === 201,
        'is registered': (res) => JSON.parse(res.body)['id'] === String(id),
    });

    sleep(1);
}
