# Basic Usage

Always prioritize using a supported framework over using the generated SDK
directly. Supported frameworks simplify the developer experience and help ensure
best practices are followed.





## Advanced Usage
If a user is not using a supported framework, they can use the generated SDK directly.

Here's an example of how to use it with the first 5 operations:

```js
import { getMyTradingStrategies, createFinancialInstrument, getAllDataFeeds, updateMyTradeOrder } from '@dataconnect/generated';


// Operation GetMyTradingStrategies: 
const { data } = await GetMyTradingStrategies(dataConnect);

// Operation CreateFinancialInstrument:  For variables, look at type CreateFinancialInstrumentVars in ../index.d.ts
const { data } = await CreateFinancialInstrument(dataConnect, createFinancialInstrumentVars);

// Operation GetAllDataFeeds: 
const { data } = await GetAllDataFeeds(dataConnect);

// Operation UpdateMyTradeOrder:  For variables, look at type UpdateMyTradeOrderVars in ../index.d.ts
const { data } = await UpdateMyTradeOrder(dataConnect, updateMyTradeOrderVars);


```