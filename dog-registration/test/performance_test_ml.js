import { check, sleep } from 'k6';
import http from 'k6/http';


export let options = {
    vus: 30, // number of virtual users
    duration: '10s', // duration of the test
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
                console.error(`registerDog: Failed to parse response body: ${e}, response: ${r.body}`);
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
                console.error(`getRegisteredDogs: Failed to parse response body: ${e}, response: ${r.body}`);
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
                console.error(`getDogById: Failed to parse response body: ${e}, response: ${r.body}`);
                return false;
            }
        },
    });
}

function convertToCSV(data) {
    const metric = data.metrics.http_req_duration;
    if (!metric || !metric.values) {
        return 'No http_req_duration data available';
    }

    const header = 'avg,min,med,max,p(90),p(95)';
    const values = [
        metric.values.avg,
        metric.values.min,
        metric.values.med,
        metric.values.max,
        metric.values['p(90)'],
        metric.values['p(95)'],
    ].join(',');

    // check if the values are correct
    console.log(header);
    console.log(values);

    return `${header}\n${values}`;
}

export function handleSummary(data) {
    return {
        'http_req_duration.csv': convertToCSV(data),
    };
}


export default function () {
    let dogId = registerDog();
    sleep(1);
    getRegisteredDogs();
    sleep(1);
    getDogById(dogId);
    sleep(1);
}
