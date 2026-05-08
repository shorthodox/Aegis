# Generated TypeScript README
This README will guide you through the process of using the generated JavaScript SDK package for the connector `example`. It will also provide examples on how to use your generated SDK to call your Data Connect queries and mutations.

***NOTE:** This README is generated alongside the generated SDK. If you make changes to this file, they will be overwritten when the SDK is regenerated.*

# Table of Contents
- [**Overview**](#generated-javascript-readme)
- [**Accessing the connector**](#accessing-the-connector)
  - [*Connecting to the local Emulator*](#connecting-to-the-local-emulator)
- [**Queries**](#queries)
  - [*GetMyTradingStrategies*](#getmytradingstrategies)
  - [*GetAllDataFeeds*](#getalldatafeeds)
- [**Mutations**](#mutations)
  - [*CreateFinancialInstrument*](#createfinancialinstrument)
  - [*UpdateMyTradeOrder*](#updatemytradeorder)

# Accessing the connector
A connector is a collection of Queries and Mutations. One SDK is generated for each connector - this SDK is generated for the connector `example`. You can find more information about connectors in the [Data Connect documentation](https://firebase.google.com/docs/data-connect#how-does).

You can use this generated SDK by importing from the package `@dataconnect/generated` as shown below. Both CommonJS and ESM imports are supported.

You can also follow the instructions from the [Data Connect documentation](https://firebase.google.com/docs/data-connect/web-sdk#set-client).

```typescript
import { getDataConnect } from 'firebase/data-connect';
import { connectorConfig } from '@dataconnect/generated';

const dataConnect = getDataConnect(connectorConfig);
```

## Connecting to the local Emulator
By default, the connector will connect to the production service.

To connect to the emulator, you can use the following code.
You can also follow the emulator instructions from the [Data Connect documentation](https://firebase.google.com/docs/data-connect/web-sdk#instrument-clients).

```typescript
import { connectDataConnectEmulator, getDataConnect } from 'firebase/data-connect';
import { connectorConfig } from '@dataconnect/generated';

const dataConnect = getDataConnect(connectorConfig);
connectDataConnectEmulator(dataConnect, 'localhost', 9399);
```

After it's initialized, you can call your Data Connect [queries](#queries) and [mutations](#mutations) from your generated SDK.

# Queries

There are two ways to execute a Data Connect Query using the generated Web SDK:
- Using a Query Reference function, which returns a `QueryRef`
  - The `QueryRef` can be used as an argument to `executeQuery()`, which will execute the Query and return a `QueryPromise`
- Using an action shortcut function, which returns a `QueryPromise`
  - Calling the action shortcut function will execute the Query and return a `QueryPromise`

The following is true for both the action shortcut function and the `QueryRef` function:
- The `QueryPromise` returned will resolve to the result of the Query once it has finished executing
- If the Query accepts arguments, both the action shortcut function and the `QueryRef` function accept a single argument: an object that contains all the required variables (and the optional variables) for the Query
- Both functions can be called with or without passing in a `DataConnect` instance as an argument. If no `DataConnect` argument is passed in, then the generated SDK will call `getDataConnect(connectorConfig)` behind the scenes for you.

Below are examples of how to use the `example` connector's generated functions to execute each query. You can also follow the examples from the [Data Connect documentation](https://firebase.google.com/docs/data-connect/web-sdk#using-queries).

## GetMyTradingStrategies
You can execute the `GetMyTradingStrategies` query using the following action shortcut function, or by calling `executeQuery()` after calling the following `QueryRef` function, both of which are defined in [dataconnect-generated/index.d.ts](./index.d.ts):
```typescript
getMyTradingStrategies(options?: ExecuteQueryOptions): QueryPromise<GetMyTradingStrategiesData, undefined>;

interface GetMyTradingStrategiesRef {
  ...
  /* Allow users to create refs without passing in DataConnect */
  (): QueryRef<GetMyTradingStrategiesData, undefined>;
}
export const getMyTradingStrategiesRef: GetMyTradingStrategiesRef;
```
You can also pass in a `DataConnect` instance to the action shortcut function or `QueryRef` function.
```typescript
getMyTradingStrategies(dc: DataConnect, options?: ExecuteQueryOptions): QueryPromise<GetMyTradingStrategiesData, undefined>;

interface GetMyTradingStrategiesRef {
  ...
  (dc: DataConnect): QueryRef<GetMyTradingStrategiesData, undefined>;
}
export const getMyTradingStrategiesRef: GetMyTradingStrategiesRef;
```

If you need the name of the operation without creating a ref, you can retrieve the operation name by calling the `operationName` property on the getMyTradingStrategiesRef:
```typescript
const name = getMyTradingStrategiesRef.operationName;
console.log(name);
```

### Variables
The `GetMyTradingStrategies` query has no variables.
### Return Type
Recall that executing the `GetMyTradingStrategies` query returns a `QueryPromise` that resolves to an object with a `data` property.

The `data` property is an object of type `GetMyTradingStrategiesData`, which is defined in [dataconnect-generated/index.d.ts](./index.d.ts). It has the following fields:
```typescript
export interface GetMyTradingStrategiesData {
  tradingStrategies: ({
    id: UUIDString;
    name: string;
    status: string;
    createdAt: TimestampString;
  } & TradingStrategy_Key)[];
}
```
### Using `GetMyTradingStrategies`'s action shortcut function

```typescript
import { getDataConnect } from 'firebase/data-connect';
import { connectorConfig, getMyTradingStrategies } from '@dataconnect/generated';


// Call the `getMyTradingStrategies()` function to execute the query.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await getMyTradingStrategies();

// You can also pass in a `DataConnect` instance to the action shortcut function.
const dataConnect = getDataConnect(connectorConfig);
const { data } = await getMyTradingStrategies(dataConnect);

console.log(data.tradingStrategies);

// Or, you can use the `Promise` API.
getMyTradingStrategies().then((response) => {
  const data = response.data;
  console.log(data.tradingStrategies);
});
```

### Using `GetMyTradingStrategies`'s `QueryRef` function

```typescript
import { getDataConnect, executeQuery } from 'firebase/data-connect';
import { connectorConfig, getMyTradingStrategiesRef } from '@dataconnect/generated';


// Call the `getMyTradingStrategiesRef()` function to get a reference to the query.
const ref = getMyTradingStrategiesRef();

// You can also pass in a `DataConnect` instance to the `QueryRef` function.
const dataConnect = getDataConnect(connectorConfig);
const ref = getMyTradingStrategiesRef(dataConnect);

// Call `executeQuery()` on the reference to execute the query.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await executeQuery(ref);

console.log(data.tradingStrategies);

// Or, you can use the `Promise` API.
executeQuery(ref).then((response) => {
  const data = response.data;
  console.log(data.tradingStrategies);
});
```

## GetAllDataFeeds
You can execute the `GetAllDataFeeds` query using the following action shortcut function, or by calling `executeQuery()` after calling the following `QueryRef` function, both of which are defined in [dataconnect-generated/index.d.ts](./index.d.ts):
```typescript
getAllDataFeeds(options?: ExecuteQueryOptions): QueryPromise<GetAllDataFeedsData, undefined>;

interface GetAllDataFeedsRef {
  ...
  /* Allow users to create refs without passing in DataConnect */
  (): QueryRef<GetAllDataFeedsData, undefined>;
}
export const getAllDataFeedsRef: GetAllDataFeedsRef;
```
You can also pass in a `DataConnect` instance to the action shortcut function or `QueryRef` function.
```typescript
getAllDataFeeds(dc: DataConnect, options?: ExecuteQueryOptions): QueryPromise<GetAllDataFeedsData, undefined>;

interface GetAllDataFeedsRef {
  ...
  (dc: DataConnect): QueryRef<GetAllDataFeedsData, undefined>;
}
export const getAllDataFeedsRef: GetAllDataFeedsRef;
```

If you need the name of the operation without creating a ref, you can retrieve the operation name by calling the `operationName` property on the getAllDataFeedsRef:
```typescript
const name = getAllDataFeedsRef.operationName;
console.log(name);
```

### Variables
The `GetAllDataFeeds` query has no variables.
### Return Type
Recall that executing the `GetAllDataFeeds` query returns a `QueryPromise` that resolves to an object with a `data` property.

The `data` property is an object of type `GetAllDataFeedsData`, which is defined in [dataconnect-generated/index.d.ts](./index.d.ts). It has the following fields:
```typescript
export interface GetAllDataFeedsData {
  dataFeeds: ({
    id: UUIDString;
    name: string;
    provider: string;
    instrumentType: string;
    connectionStatus: string;
  } & DataFeed_Key)[];
}
```
### Using `GetAllDataFeeds`'s action shortcut function

```typescript
import { getDataConnect } from 'firebase/data-connect';
import { connectorConfig, getAllDataFeeds } from '@dataconnect/generated';


// Call the `getAllDataFeeds()` function to execute the query.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await getAllDataFeeds();

// You can also pass in a `DataConnect` instance to the action shortcut function.
const dataConnect = getDataConnect(connectorConfig);
const { data } = await getAllDataFeeds(dataConnect);

console.log(data.dataFeeds);

// Or, you can use the `Promise` API.
getAllDataFeeds().then((response) => {
  const data = response.data;
  console.log(data.dataFeeds);
});
```

### Using `GetAllDataFeeds`'s `QueryRef` function

```typescript
import { getDataConnect, executeQuery } from 'firebase/data-connect';
import { connectorConfig, getAllDataFeedsRef } from '@dataconnect/generated';


// Call the `getAllDataFeedsRef()` function to get a reference to the query.
const ref = getAllDataFeedsRef();

// You can also pass in a `DataConnect` instance to the `QueryRef` function.
const dataConnect = getDataConnect(connectorConfig);
const ref = getAllDataFeedsRef(dataConnect);

// Call `executeQuery()` on the reference to execute the query.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await executeQuery(ref);

console.log(data.dataFeeds);

// Or, you can use the `Promise` API.
executeQuery(ref).then((response) => {
  const data = response.data;
  console.log(data.dataFeeds);
});
```

# Mutations

There are two ways to execute a Data Connect Mutation using the generated Web SDK:
- Using a Mutation Reference function, which returns a `MutationRef`
  - The `MutationRef` can be used as an argument to `executeMutation()`, which will execute the Mutation and return a `MutationPromise`
- Using an action shortcut function, which returns a `MutationPromise`
  - Calling the action shortcut function will execute the Mutation and return a `MutationPromise`

The following is true for both the action shortcut function and the `MutationRef` function:
- The `MutationPromise` returned will resolve to the result of the Mutation once it has finished executing
- If the Mutation accepts arguments, both the action shortcut function and the `MutationRef` function accept a single argument: an object that contains all the required variables (and the optional variables) for the Mutation
- Both functions can be called with or without passing in a `DataConnect` instance as an argument. If no `DataConnect` argument is passed in, then the generated SDK will call `getDataConnect(connectorConfig)` behind the scenes for you.

Below are examples of how to use the `example` connector's generated functions to execute each mutation. You can also follow the examples from the [Data Connect documentation](https://firebase.google.com/docs/data-connect/web-sdk#using-mutations).

## CreateFinancialInstrument
You can execute the `CreateFinancialInstrument` mutation using the following action shortcut function, or by calling `executeMutation()` after calling the following `MutationRef` function, both of which are defined in [dataconnect-generated/index.d.ts](./index.d.ts):
```typescript
createFinancialInstrument(vars: CreateFinancialInstrumentVariables): MutationPromise<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;

interface CreateFinancialInstrumentRef {
  ...
  /* Allow users to create refs without passing in DataConnect */
  (vars: CreateFinancialInstrumentVariables): MutationRef<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;
}
export const createFinancialInstrumentRef: CreateFinancialInstrumentRef;
```
You can also pass in a `DataConnect` instance to the action shortcut function or `MutationRef` function.
```typescript
createFinancialInstrument(dc: DataConnect, vars: CreateFinancialInstrumentVariables): MutationPromise<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;

interface CreateFinancialInstrumentRef {
  ...
  (dc: DataConnect, vars: CreateFinancialInstrumentVariables): MutationRef<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;
}
export const createFinancialInstrumentRef: CreateFinancialInstrumentRef;
```

If you need the name of the operation without creating a ref, you can retrieve the operation name by calling the `operationName` property on the createFinancialInstrumentRef:
```typescript
const name = createFinancialInstrumentRef.operationName;
console.log(name);
```

### Variables
The `CreateFinancialInstrument` mutation requires an argument of type `CreateFinancialInstrumentVariables`, which is defined in [dataconnect-generated/index.d.ts](./index.d.ts). It has the following fields:

```typescript
export interface CreateFinancialInstrumentVariables {
  symbol: string;
  name: string;
  type: string;
  exchange: string;
  description?: string | null;
  marketHours?: string | null;
}
```
### Return Type
Recall that executing the `CreateFinancialInstrument` mutation returns a `MutationPromise` that resolves to an object with a `data` property.

The `data` property is an object of type `CreateFinancialInstrumentData`, which is defined in [dataconnect-generated/index.d.ts](./index.d.ts). It has the following fields:
```typescript
export interface CreateFinancialInstrumentData {
  financialInstrument_insert: FinancialInstrument_Key;
}
```
### Using `CreateFinancialInstrument`'s action shortcut function

```typescript
import { getDataConnect } from 'firebase/data-connect';
import { connectorConfig, createFinancialInstrument, CreateFinancialInstrumentVariables } from '@dataconnect/generated';

// The `CreateFinancialInstrument` mutation requires an argument of type `CreateFinancialInstrumentVariables`:
const createFinancialInstrumentVars: CreateFinancialInstrumentVariables = {
  symbol: ..., 
  name: ..., 
  type: ..., 
  exchange: ..., 
  description: ..., // optional
  marketHours: ..., // optional
};

// Call the `createFinancialInstrument()` function to execute the mutation.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await createFinancialInstrument(createFinancialInstrumentVars);
// Variables can be defined inline as well.
const { data } = await createFinancialInstrument({ symbol: ..., name: ..., type: ..., exchange: ..., description: ..., marketHours: ..., });

// You can also pass in a `DataConnect` instance to the action shortcut function.
const dataConnect = getDataConnect(connectorConfig);
const { data } = await createFinancialInstrument(dataConnect, createFinancialInstrumentVars);

console.log(data.financialInstrument_insert);

// Or, you can use the `Promise` API.
createFinancialInstrument(createFinancialInstrumentVars).then((response) => {
  const data = response.data;
  console.log(data.financialInstrument_insert);
});
```

### Using `CreateFinancialInstrument`'s `MutationRef` function

```typescript
import { getDataConnect, executeMutation } from 'firebase/data-connect';
import { connectorConfig, createFinancialInstrumentRef, CreateFinancialInstrumentVariables } from '@dataconnect/generated';

// The `CreateFinancialInstrument` mutation requires an argument of type `CreateFinancialInstrumentVariables`:
const createFinancialInstrumentVars: CreateFinancialInstrumentVariables = {
  symbol: ..., 
  name: ..., 
  type: ..., 
  exchange: ..., 
  description: ..., // optional
  marketHours: ..., // optional
};

// Call the `createFinancialInstrumentRef()` function to get a reference to the mutation.
const ref = createFinancialInstrumentRef(createFinancialInstrumentVars);
// Variables can be defined inline as well.
const ref = createFinancialInstrumentRef({ symbol: ..., name: ..., type: ..., exchange: ..., description: ..., marketHours: ..., });

// You can also pass in a `DataConnect` instance to the `MutationRef` function.
const dataConnect = getDataConnect(connectorConfig);
const ref = createFinancialInstrumentRef(dataConnect, createFinancialInstrumentVars);

// Call `executeMutation()` on the reference to execute the mutation.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await executeMutation(ref);

console.log(data.financialInstrument_insert);

// Or, you can use the `Promise` API.
executeMutation(ref).then((response) => {
  const data = response.data;
  console.log(data.financialInstrument_insert);
});
```

## UpdateMyTradeOrder
You can execute the `UpdateMyTradeOrder` mutation using the following action shortcut function, or by calling `executeMutation()` after calling the following `MutationRef` function, both of which are defined in [dataconnect-generated/index.d.ts](./index.d.ts):
```typescript
updateMyTradeOrder(vars: UpdateMyTradeOrderVariables): MutationPromise<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;

interface UpdateMyTradeOrderRef {
  ...
  /* Allow users to create refs without passing in DataConnect */
  (vars: UpdateMyTradeOrderVariables): MutationRef<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;
}
export const updateMyTradeOrderRef: UpdateMyTradeOrderRef;
```
You can also pass in a `DataConnect` instance to the action shortcut function or `MutationRef` function.
```typescript
updateMyTradeOrder(dc: DataConnect, vars: UpdateMyTradeOrderVariables): MutationPromise<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;

interface UpdateMyTradeOrderRef {
  ...
  (dc: DataConnect, vars: UpdateMyTradeOrderVariables): MutationRef<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;
}
export const updateMyTradeOrderRef: UpdateMyTradeOrderRef;
```

If you need the name of the operation without creating a ref, you can retrieve the operation name by calling the `operationName` property on the updateMyTradeOrderRef:
```typescript
const name = updateMyTradeOrderRef.operationName;
console.log(name);
```

### Variables
The `UpdateMyTradeOrder` mutation requires an argument of type `UpdateMyTradeOrderVariables`, which is defined in [dataconnect-generated/index.d.ts](./index.d.ts). It has the following fields:

```typescript
export interface UpdateMyTradeOrderVariables {
  orderId: string;
  status: string;
}
```
### Return Type
Recall that executing the `UpdateMyTradeOrder` mutation returns a `MutationPromise` that resolves to an object with a `data` property.

The `data` property is an object of type `UpdateMyTradeOrderData`, which is defined in [dataconnect-generated/index.d.ts](./index.d.ts). It has the following fields:
```typescript
export interface UpdateMyTradeOrderData {
  tradeOrder_updateMany: number;
}
```
### Using `UpdateMyTradeOrder`'s action shortcut function

```typescript
import { getDataConnect } from 'firebase/data-connect';
import { connectorConfig, updateMyTradeOrder, UpdateMyTradeOrderVariables } from '@dataconnect/generated';

// The `UpdateMyTradeOrder` mutation requires an argument of type `UpdateMyTradeOrderVariables`:
const updateMyTradeOrderVars: UpdateMyTradeOrderVariables = {
  orderId: ..., 
  status: ..., 
};

// Call the `updateMyTradeOrder()` function to execute the mutation.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await updateMyTradeOrder(updateMyTradeOrderVars);
// Variables can be defined inline as well.
const { data } = await updateMyTradeOrder({ orderId: ..., status: ..., });

// You can also pass in a `DataConnect` instance to the action shortcut function.
const dataConnect = getDataConnect(connectorConfig);
const { data } = await updateMyTradeOrder(dataConnect, updateMyTradeOrderVars);

console.log(data.tradeOrder_updateMany);

// Or, you can use the `Promise` API.
updateMyTradeOrder(updateMyTradeOrderVars).then((response) => {
  const data = response.data;
  console.log(data.tradeOrder_updateMany);
});
```

### Using `UpdateMyTradeOrder`'s `MutationRef` function

```typescript
import { getDataConnect, executeMutation } from 'firebase/data-connect';
import { connectorConfig, updateMyTradeOrderRef, UpdateMyTradeOrderVariables } from '@dataconnect/generated';

// The `UpdateMyTradeOrder` mutation requires an argument of type `UpdateMyTradeOrderVariables`:
const updateMyTradeOrderVars: UpdateMyTradeOrderVariables = {
  orderId: ..., 
  status: ..., 
};

// Call the `updateMyTradeOrderRef()` function to get a reference to the mutation.
const ref = updateMyTradeOrderRef(updateMyTradeOrderVars);
// Variables can be defined inline as well.
const ref = updateMyTradeOrderRef({ orderId: ..., status: ..., });

// You can also pass in a `DataConnect` instance to the `MutationRef` function.
const dataConnect = getDataConnect(connectorConfig);
const ref = updateMyTradeOrderRef(dataConnect, updateMyTradeOrderVars);

// Call `executeMutation()` on the reference to execute the mutation.
// You can use the `await` keyword to wait for the promise to resolve.
const { data } = await executeMutation(ref);

console.log(data.tradeOrder_updateMany);

// Or, you can use the `Promise` API.
executeMutation(ref).then((response) => {
  const data = response.data;
  console.log(data.tradeOrder_updateMany);
});
```

