# Edge Score Distribution

## Distribution Summary Statistics
- **Min:** 19.6867
- **Max:** 100.0000
- **Mean:** 56.2845
- **Std Dev:** 13.2254

## Percentiles
- **P50:** 56.8605
- **P60:** 60.9503
- **P70:** 64.3454
- **P80:** 66.9466
- **P90:** 71.0468
- **P95:** 78.1047
- **P99:** 86.3644

### Audit Conclusion
The Edge Score is NOT compressed. The maximum score reaches 100.0 and the 80th percentile is around 65.0. Therefore, the threshold of 55.0 is highly reachable. The reason for 0 trades in the user's old run was the exclusive combination of the Sell-Blocking bug and overrestrictive filters vetoing all signals.