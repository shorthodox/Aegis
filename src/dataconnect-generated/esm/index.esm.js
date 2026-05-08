import { queryRef, executeQuery, validateArgsWithOptions, mutationRef, executeMutation, validateArgs, makeMemoryCacheProvider } from 'firebase/data-connect';

export const connectorConfig = {
  connector: 'example',
  service: 'aisignalbot',
  location: 'us-east4'
};
export const dataConnectSettings = {
  cacheSettings: {
    cacheProvider: makeMemoryCacheProvider()
  }
};
export const getMyTradingStrategiesRef = (dc) => {
  const { dc: dcInstance} = validateArgs(connectorConfig, dc, undefined);
  dcInstance._useGeneratedSdk();
  return queryRef(dcInstance, 'GetMyTradingStrategies');
}
getMyTradingStrategiesRef.operationName = 'GetMyTradingStrategies';

export function getMyTradingStrategies(dcOrOptions, options) {
  
  const { dc: dcInstance, vars: inputVars, options: inputOpts } = validateArgsWithOptions(connectorConfig, dcOrOptions, options, undefined,false, false);
  return executeQuery(getMyTradingStrategiesRef(dcInstance, inputVars), inputOpts && inputOpts.fetchPolicy);
}

export const createFinancialInstrumentRef = (dcOrVars, vars) => {
  const { dc: dcInstance, vars: inputVars} = validateArgs(connectorConfig, dcOrVars, vars, true);
  dcInstance._useGeneratedSdk();
  return mutationRef(dcInstance, 'CreateFinancialInstrument', inputVars);
}
createFinancialInstrumentRef.operationName = 'CreateFinancialInstrument';

export function createFinancialInstrument(dcOrVars, vars) {
  const { dc: dcInstance, vars: inputVars } = validateArgs(connectorConfig, dcOrVars, vars, true);
  return executeMutation(createFinancialInstrumentRef(dcInstance, inputVars));
}

export const getAllDataFeedsRef = (dc) => {
  const { dc: dcInstance} = validateArgs(connectorConfig, dc, undefined);
  dcInstance._useGeneratedSdk();
  return queryRef(dcInstance, 'GetAllDataFeeds');
}
getAllDataFeedsRef.operationName = 'GetAllDataFeeds';

export function getAllDataFeeds(dcOrOptions, options) {
  
  const { dc: dcInstance, vars: inputVars, options: inputOpts } = validateArgsWithOptions(connectorConfig, dcOrOptions, options, undefined,false, false);
  return executeQuery(getAllDataFeedsRef(dcInstance, inputVars), inputOpts && inputOpts.fetchPolicy);
}

export const updateMyTradeOrderRef = (dcOrVars, vars) => {
  const { dc: dcInstance, vars: inputVars} = validateArgs(connectorConfig, dcOrVars, vars, true);
  dcInstance._useGeneratedSdk();
  return mutationRef(dcInstance, 'UpdateMyTradeOrder', inputVars);
}
updateMyTradeOrderRef.operationName = 'UpdateMyTradeOrder';

export function updateMyTradeOrder(dcOrVars, vars) {
  const { dc: dcInstance, vars: inputVars } = validateArgs(connectorConfig, dcOrVars, vars, true);
  return executeMutation(updateMyTradeOrderRef(dcInstance, inputVars));
}

