const { queryRef, executeQuery, validateArgsWithOptions, mutationRef, executeMutation, validateArgs, makeMemoryCacheProvider } = require('firebase/data-connect');

const connectorConfig = {
  connector: 'example',
  service: 'aisignalbot',
  location: 'us-east4'
};
exports.connectorConfig = connectorConfig;
const dataConnectSettings = {
  cacheSettings: {
    cacheProvider: makeMemoryCacheProvider()
  }
};
exports.dataConnectSettings = dataConnectSettings;

const getMyTradingStrategiesRef = (dc) => {
  const { dc: dcInstance} = validateArgs(connectorConfig, dc, undefined);
  dcInstance._useGeneratedSdk();
  return queryRef(dcInstance, 'GetMyTradingStrategies');
}
getMyTradingStrategiesRef.operationName = 'GetMyTradingStrategies';
exports.getMyTradingStrategiesRef = getMyTradingStrategiesRef;

exports.getMyTradingStrategies = function getMyTradingStrategies(dcOrOptions, options) {
  
  const { dc: dcInstance, vars: inputVars, options: inputOpts } = validateArgsWithOptions(connectorConfig, dcOrOptions, options, undefined,false, false);
  return executeQuery(getMyTradingStrategiesRef(dcInstance, inputVars), inputOpts && inputOpts.fetchPolicy);
}
;

const createFinancialInstrumentRef = (dcOrVars, vars) => {
  const { dc: dcInstance, vars: inputVars} = validateArgs(connectorConfig, dcOrVars, vars, true);
  dcInstance._useGeneratedSdk();
  return mutationRef(dcInstance, 'CreateFinancialInstrument', inputVars);
}
createFinancialInstrumentRef.operationName = 'CreateFinancialInstrument';
exports.createFinancialInstrumentRef = createFinancialInstrumentRef;

exports.createFinancialInstrument = function createFinancialInstrument(dcOrVars, vars) {
  const { dc: dcInstance, vars: inputVars } = validateArgs(connectorConfig, dcOrVars, vars, true);
  return executeMutation(createFinancialInstrumentRef(dcInstance, inputVars));
}
;

const getAllDataFeedsRef = (dc) => {
  const { dc: dcInstance} = validateArgs(connectorConfig, dc, undefined);
  dcInstance._useGeneratedSdk();
  return queryRef(dcInstance, 'GetAllDataFeeds');
}
getAllDataFeedsRef.operationName = 'GetAllDataFeeds';
exports.getAllDataFeedsRef = getAllDataFeedsRef;

exports.getAllDataFeeds = function getAllDataFeeds(dcOrOptions, options) {
  
  const { dc: dcInstance, vars: inputVars, options: inputOpts } = validateArgsWithOptions(connectorConfig, dcOrOptions, options, undefined,false, false);
  return executeQuery(getAllDataFeedsRef(dcInstance, inputVars), inputOpts && inputOpts.fetchPolicy);
}
;

const updateMyTradeOrderRef = (dcOrVars, vars) => {
  const { dc: dcInstance, vars: inputVars} = validateArgs(connectorConfig, dcOrVars, vars, true);
  dcInstance._useGeneratedSdk();
  return mutationRef(dcInstance, 'UpdateMyTradeOrder', inputVars);
}
updateMyTradeOrderRef.operationName = 'UpdateMyTradeOrder';
exports.updateMyTradeOrderRef = updateMyTradeOrderRef;

exports.updateMyTradeOrder = function updateMyTradeOrder(dcOrVars, vars) {
  const { dc: dcInstance, vars: inputVars } = validateArgs(connectorConfig, dcOrVars, vars, true);
  return executeMutation(updateMyTradeOrderRef(dcInstance, inputVars));
}
;
