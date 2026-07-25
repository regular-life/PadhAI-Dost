import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom Metrics
export const cacheHitRate = new Rate('semantic_cache_hit_rate');
export const cacheHitLatency = new Trend('semantic_cache_hit_latency_ms');
export const cacheMissLatency = new Trend('semantic_cache_miss_latency_ms');

export const options = {
  stages: [
    { duration: '10s', target: 5 },  // Ramp-up to 5 Virtual Users (VUs)
    { duration: '30s', target: 20 }, // Sustain 20 VUs
    { duration: '10s', target: 0 },  // Ramp-down to 0
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],              // Less than 1% errors
    http_req_duration: ['p(95)<2000'],           // p95 latency under 2000ms
    semantic_cache_hit_rate: ['rate>0.70'],     // Cache hit rate > 70% for repeated queries
    semantic_cache_hit_latency_ms: ['p(95)<50'],// Cache hit p95 latency under 50ms
  },
};

const BASE_URL = __ENV.API_BASE || 'http://localhost:8080';
const DOC_ID = __ENV.DOC_ID || 'doc_wgan_123';

// Sample query clusters (canonical query + semantically similar variants)
const QUERY_VARIANTS = [
  'What is the Wasserstein distance in WGAN?',
  'Explain how Wasserstein distance improves GAN stability',
  'Why is Earth Mover distance preferred over JS divergence?',
  'What is the role of the critic network in WGAN?',
  'How does the critic network differ from a standard discriminator?',
];

export function setup() {
  // Login to acquire JWT Auth Token
  const loginRes = http.post(`${BASE_URL}/api/v1/auth/login`, JSON.stringify({
    username: 'k6_tester',
    password: 'password123',
  }), {
    headers: { 'Content-Type': 'application/json' },
  });

  let token = '';
  if (loginRes.status === 200) {
    token = loginRes.json('token');
  }

  return { token: token };
}

export default function (data) {
  const queryIndex = Math.floor(Math.random() * QUERY_VARIANTS.length);
  const question = QUERY_VARIANTS[queryIndex];

  const payload = JSON.stringify({
    question: question,
    doc_id: DOC_ID,
    top_k: 5,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': data.token ? `Bearer ${data.token}` : '',
    },
  };

  const startTime = Date.now();
  const res = http.post(`${BASE_URL}/api/v1/query`, payload, params);
  const duration = Date.now() - startTime;

  const isSuccess = check(res, {
    'status is 200': (r) => r.status === 200,
    'has answer': (r) => r.json('answer') !== undefined,
  });

  if (isSuccess && res.status === 200) {
    const isCacheHit = res.json('cache_hit') === true;
    cacheHitRate.add(isCacheHit);

    if (isCacheHit) {
      cacheHitLatency.add(duration);
    } else {
      cacheMissLatency.add(duration);
    }
  }

  sleep(0.5); // 500ms pacing between iterations
}
