// razorpay-checkout.js
// Frontend helper to create order and open Razorpay standard checkout

async function fetchKeyId() {
  const resp = await fetch('/api/razorpay-key');
  if (!resp.ok) throw new Error('failed to fetch key id');
  const j = await resp.json();
  return j.key_id;
}

async function createOrder(amountPaise) {
  const resp = await fetch('/api/create-order', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount: amountPaise }),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error('create-order failed: ' + txt);
  }
  return resp.json();
}

async function verifyPayment(order_id, payment_id, signature) {
  const resp = await fetch('/api/verify-payment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      razorpay_order_id: order_id,
      razorpay_payment_id: payment_id,
      razorpay_signature: signature,
    }),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error('verify failed: ' + txt);
  }
  return resp.json();
}

// Map plan slugs to amounts (paise)
const PLAN_PRICES = {
  basic: 590, // $5.90 -> approximate in cents/paise; you should adjust to INR values
  intermediate: 2400,
  pro: 4000,
};

document.addEventListener('click', async (ev) => {
  const el = ev.target.closest('[data-plan]');
  if (!el) return;
  ev.preventDefault();
  try {
    const plan = el.dataset.plan;
    const amount = PLAN_PRICES[plan] || 1000;
    const keyId = await fetchKeyId();
    const order = await createOrder(amount);
    const options = {
      key: keyId,
      amount: order.amount,
      currency: order.currency,
      name: 'AEGIS',
      description: `Subscription ${plan}`,
      order_id: order.order_id,
      handler: async function (response) {
        try {
          await verifyPayment(response.razorpay_order_id, response.razorpay_payment_id, response.razorpay_signature);
          alert('Payment verified. Thank you!');
          // TODO: redirect to success page / activate subscription
        } catch (err) {
          alert('Payment verification failed: ' + err.message);
        }
      },
      modal: {
        ondismiss: function () { alert('Payment cancelled'); }
      }
    };
    const rzp = new Razorpay(options);
    rzp.open();
  } catch (err) {
    console.error(err);
    alert('Payment failed: ' + (err.message || err));
  }
}, false);
