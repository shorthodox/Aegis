#!/usr/bin/env python
"""Provision the AEGIS product catalog in Paddle Billing.

Creates one product + one monthly recurring price per plan tier and prints the
PADDLE_PRICE_ID_* lines to paste into .env.

SAFETY
------
Dry run by default: it shows exactly what it would create and writes nothing.
Creating in the LIVE account needs BOTH --apply and --live, because Paddle
products and prices cannot be deleted — only archived. A mistake here is
permanent clutter in a real billing account, and a wrong price is a wrong
charge to a real customer.

Idempotent: every product carries custom_data.aegis_plan, and a re-run reuses
anything already tagged rather than creating duplicates.

Usage
-----
    # see the plan, touch nothing (sandbox key)
    python -m scripts.paddle_provision

    # create in sandbox
    PADDLE_API_KEY=pdl_sdbx_apikey_... python -m scripts.paddle_provision --apply

    # create in LIVE — deliberate, two flags
    PADDLE_API_KEY=pdl_live_apikey_... python -m scripts.paddle_provision --apply --live

Notes
-----
* tax_category has real tax consequences and is NOT a safe default to guess.
  It ships as 'standard'; confirm with whoever does your accounting before
  running against live. Override with --tax-category (see CATEGORIES).
* The 7-day refund is NOT set here. Paddle has no refund-window field on a
  product or price — it is policy, stated in your terms and applied when you
  issue the refund. Do not model it as trial_period: a trial means "free for
  7 days", which is a different product from "pay now, refundable for 7 days".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CATEGORIES = (
    'standard', 'saas', 'digital-goods', 'ebooks', 'implementation-services',
    'professional-services', 'software-programming-services',
    'training-services', 'website-hosting',
)

# Customer-facing name -> internal tier key.
#
# The internal keys are NOT renamed. They are written into every Firestore user
# document and read by the plan gates, so renaming them would need a data
# migration. Note the collision this creates and do not "tidy" it away:
# the customer-facing "Pro" is the MIDDLE tier and maps to internal
# 'intermediate'; internal 'pro' is the TOP tier, sold as "Advanced".
PLANS: List[Dict[str, Any]] = [
    {
        'tier': 'basic',
        'name': 'Starter',
        'usd': 5.90,
        'blurb': 'AEGIS signal access — entry tier.',
    },
    {
        'tier': 'intermediate',
        'name': 'Pro',
        'usd': 14.00,
        'blurb': 'AEGIS signal access — full fleet coverage.',
    },
    {
        'tier': 'pro',
        'name': 'Advanced',
        'usd': 30.00,
        'blurb': 'AEGIS signal access — full fleet, priority alerts.',
    },
]


def _minor_units(usd: float) -> str:
    """USD amount -> Paddle's string of minor units. 5.90 -> '590'."""
    return str(int(round(usd * 100)))


class Paddle:
    def __init__(self, key: str, live: bool, timeout: float = 20.0):
        self.base = ('https://api.paddle.com' if live
                     else 'https://sandbox-api.paddle.com')
        self._c = httpx.Client(
            timeout=timeout,
            headers={'Authorization': f'Bearer {key}',
                     'Content-Type': 'application/json'},
        )

    def get(self, path: str, **params) -> List[Dict[str, Any]]:
        r = self._c.get(f'{self.base}{path}', params=params or None)
        r.raise_for_status()
        return r.json().get('data') or []

    def post(self, path: str, payload: dict) -> Dict[str, Any]:
        r = self._c.post(f'{self.base}{path}', json=payload)
        if r.status_code >= 400:
            raise SystemExit(f'Paddle {r.status_code} on POST {path}\n'
                             f'  request : {json.dumps(payload)}\n'
                             f'  response: {r.text}')
        return r.json().get('data') or {}

    def close(self) -> None:
        self._c.close()


def _find_product(products: List[Dict[str, Any]], tier: str) -> Optional[Dict[str, Any]]:
    for p in products:
        if ((p.get('custom_data') or {}).get('aegis_plan')) == tier:
            return p
    return None


def _find_price(prices: List[Dict[str, Any]], product_id: str,
                amount: str) -> Optional[Dict[str, Any]]:
    for pr in prices:
        if pr.get('product_id') != product_id:
            continue
        up = pr.get('unit_price') or {}
        bc = pr.get('billing_cycle') or {}
        if (up.get('amount') == amount and up.get('currency_code') == 'USD'
                and bc.get('interval') == 'month' and bc.get('frequency') == 1):
            return pr
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='actually create (default is a dry run)')
    ap.add_argument('--live', action='store_true',
                    help='target the LIVE account instead of sandbox')
    ap.add_argument('--tax-category', default='standard', choices=CATEGORIES,
                    help="Paddle tax category (default: standard) — confirm this")
    args = ap.parse_args()

    key = os.getenv('PADDLE_API_KEY', '').strip()

    print(f'AEGIS -> Paddle catalog  [{"LIVE" if args.live else "SANDBOX"}]'
          f'{"" if args.apply else "   (DRY RUN — nothing will be created)"}')
    print(f'tax_category = {args.tax_category}\n')
    print(f'{"customer-facing":<18}{"internal tier":<15}{"price":>9}   billing')
    for p in PLANS:
        print(f'{p["name"]:<18}{p["tier"]:<15}'
              f'{"$" + format(p["usd"], ".2f"):>9}   monthly, USD')
    print()
    print('NOT configured here: the 7-day refund. Paddle has no refund-window')
    print('field on a product or price — it is policy (already stated on')
    print('/refund-policy) and is applied when you issue the refund.\n')

    if not args.apply:
        print('Dry run. Re-run with --apply (plus --live for the live account).')
        return 0

    if not key:
        print('PADDLE_API_KEY is not set — cannot create anything.', file=sys.stderr)
        return 2
    if args.live and not key.startswith('pdl_live_'):
        print(f'--live given but the key is not a live key (starts '
              f'{key[:9]!r}). Refusing.', file=sys.stderr)
        return 2
    if not args.live and key.startswith('pdl_live_'):
        print('A LIVE key was supplied without --live. Refusing rather than '
              'quietly writing to production.', file=sys.stderr)
        return 2

    if args.live:
        print('About to create products and prices in the LIVE Paddle account.')
        print('Paddle entities cannot be deleted, only archived.')
        if input('Type "create live" to continue: ').strip() != 'create live':
            print('Aborted.')
            return 1
        print()

    api = Paddle(key, live=args.live)
    try:
        products = api.get('/products', status='active', per_page=200)
        prices = api.get('/prices', status='active', per_page=200)

        env_lines: List[str] = []
        for plan in PLANS:
            tier, name, usd = plan['tier'], plan['name'], plan['usd']
            amount = _minor_units(usd)

            prod = _find_product(products, tier)
            if prod:
                print(f'= product exists  {name:<10} {prod["id"]}')
            else:
                prod = api.post('/products', {
                    'name': f'AEGIS {name}',
                    'tax_category': args.tax_category,
                    'description': plan['blurb'],
                    'custom_data': {'aegis_plan': tier},
                })
                print(f'+ product created {name:<10} {prod["id"]}')

            price = _find_price(prices, prod['id'], amount)
            if price:
                print(f'= price exists    {name:<10} {price["id"]}  ${usd:.2f}/mo')
            else:
                price = api.post('/prices', {
                    'product_id': prod['id'],
                    'name': f'{name} Monthly',
                    'description': f'AEGIS {name} — USD {usd:.2f}/month',
                    'unit_price': {'amount': amount, 'currency_code': 'USD'},
                    'billing_cycle': {'interval': 'month', 'frequency': 1},
                    'custom_data': {'aegis_plan': tier},
                })
                print(f'+ price created   {name:<10} {price["id"]}  ${usd:.2f}/mo')

            env_lines.append(f'PADDLE_PRICE_ID_{tier.upper()}="{price["id"]}"')

        print('\nAdd to .env:\n')
        print(f'PADDLE_API_KEY="{key[:14]}..."   # the key you just used')
        print(f'PADDLE_MODE="{"live" if args.live else "sandbox"}"')
        for line in env_lines:
            print(line)
        print('PADDLE_WEBHOOK_SECRET="pdl_ntfset_..."   '
              '# Developer tools > Notifications')
        print('\nAlso set a default payment link under Paddle > Checkout settings,')
        print('or /transactions returns no checkout URL.')
    finally:
        api.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
