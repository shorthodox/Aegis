import { ConnectorConfig, DataConnect, QueryRef, QueryPromise, ExecuteQueryOptions, MutationRef, MutationPromise, DataConnectSettings } from 'firebase/data-connect';

export const connectorConfig: ConnectorConfig;
export const dataConnectSettings: DataConnectSettings;

export type TimestampString = string;
export type UUIDString = string;
export type Int64String = string;
export type DateString = string;




export interface BacktestResult_Key {
  id: UUIDString;
  __typename?: 'BacktestResult_Key';
}

export interface CreateFinancialInstrumentData {
  financialInstrument_insert: FinancialInstrument_Key;
}

export interface CreateFinancialInstrumentVariables {
  symbol: string;
  name: string;
  type: string;
  exchange: string;
  description?: string | null;
  marketHours?: string | null;
}

export interface DataFeed_Key {
  id: UUIDString;
  __typename?: 'DataFeed_Key';
}

export interface FinancialInstrument_Key {
  id: UUIDString;
  __typename?: 'FinancialInstrument_Key';
}

export interface GetAllDataFeedsData {
  dataFeeds: ({
    id: UUIDString;
    name: string;
    provider: string;
    instrumentType: string;
    connectionStatus: string;
  } & DataFeed_Key)[];
}

export interface GetMyTradingStrategiesData {
  tradingStrategies: ({
    id: UUIDString;
    name: string;
    status: string;
    createdAt: TimestampString;
  } & TradingStrategy_Key)[];
}

export interface TradeOrder_Key {
  id: UUIDString;
  __typename?: 'TradeOrder_Key';
}

export interface TradingStrategy_Key {
  id: UUIDString;
  __typename?: 'TradingStrategy_Key';
}

export interface UpdateMyTradeOrderData {
  tradeOrder_updateMany: number;
}

export interface UpdateMyTradeOrderVariables {
  orderId: string;
  status: string;
}

export interface User_Key {
  id: UUIDString;
  __typename?: 'User_Key';
}

interface GetMyTradingStrategiesRef {
  /* Allow users to create refs without passing in DataConnect */
  (): QueryRef<GetMyTradingStrategiesData, undefined>;
  /* Allow users to pass in custom DataConnect instances */
  (dc: DataConnect): QueryRef<GetMyTradingStrategiesData, undefined>;
  operationName: string;
}
export const getMyTradingStrategiesRef: GetMyTradingStrategiesRef;

export function getMyTradingStrategies(options?: ExecuteQueryOptions): QueryPromise<GetMyTradingStrategiesData, undefined>;
export function getMyTradingStrategies(dc: DataConnect, options?: ExecuteQueryOptions): QueryPromise<GetMyTradingStrategiesData, undefined>;

interface CreateFinancialInstrumentRef {
  /* Allow users to create refs without passing in DataConnect */
  (vars: CreateFinancialInstrumentVariables): MutationRef<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;
  /* Allow users to pass in custom DataConnect instances */
  (dc: DataConnect, vars: CreateFinancialInstrumentVariables): MutationRef<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;
  operationName: string;
}
export const createFinancialInstrumentRef: CreateFinancialInstrumentRef;

export function createFinancialInstrument(vars: CreateFinancialInstrumentVariables): MutationPromise<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;
export function createFinancialInstrument(dc: DataConnect, vars: CreateFinancialInstrumentVariables): MutationPromise<CreateFinancialInstrumentData, CreateFinancialInstrumentVariables>;

interface GetAllDataFeedsRef {
  /* Allow users to create refs without passing in DataConnect */
  (): QueryRef<GetAllDataFeedsData, undefined>;
  /* Allow users to pass in custom DataConnect instances */
  (dc: DataConnect): QueryRef<GetAllDataFeedsData, undefined>;
  operationName: string;
}
export const getAllDataFeedsRef: GetAllDataFeedsRef;

export function getAllDataFeeds(options?: ExecuteQueryOptions): QueryPromise<GetAllDataFeedsData, undefined>;
export function getAllDataFeeds(dc: DataConnect, options?: ExecuteQueryOptions): QueryPromise<GetAllDataFeedsData, undefined>;

interface UpdateMyTradeOrderRef {
  /* Allow users to create refs without passing in DataConnect */
  (vars: UpdateMyTradeOrderVariables): MutationRef<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;
  /* Allow users to pass in custom DataConnect instances */
  (dc: DataConnect, vars: UpdateMyTradeOrderVariables): MutationRef<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;
  operationName: string;
}
export const updateMyTradeOrderRef: UpdateMyTradeOrderRef;

export function updateMyTradeOrder(vars: UpdateMyTradeOrderVariables): MutationPromise<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;
export function updateMyTradeOrder(dc: DataConnect, vars: UpdateMyTradeOrderVariables): MutationPromise<UpdateMyTradeOrderData, UpdateMyTradeOrderVariables>;

