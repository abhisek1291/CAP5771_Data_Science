import pandas as pd
import numpy as np
import math
from datetime import datetime

count = 0
flow_cols = []


def predict_flow_by_detector(result, columns_flow):
    global flow_cols
    flow_cols = columns_flow
    print('\nstarting method 2 predict')
    all_flows = []
    for i in range(0, len(columns_flow), 1):
        all_flows.append(result[result.detector == str("x" + str(i))].reset_index(drop=True))

    results = []
    confidences = []
    for i in range(0, len(columns_flow), 1):
        result_x = all_flows[i].sort_values("timestamp")
        result_x = result_x.reset_index(drop=True)
        result_x['timestamp'] = pd.to_datetime(result_x['timestamp'])
        timestamp = result_x["timestamp"]
        timestamp = np.asarray(timestamp)
        flow_x1 = result_x["flow"]
        flow_x1 = np.asarray(flow_x1)
        density = result_x["probability"]
        density = np.asarray(density)
        resultant, confidence = predict_flow(timestamp, flow_x1, density, i)
        results.append(resultant)
        confidences.append(confidence)

    for i in range(0, len(columns_flow), 1):
        all_flows[i]["Expected_" + str(i+1)] = pd.DataFrame(results[i])
        all_flows[i]["Confidence_" + str(i+1)] = pd.DataFrame(confidences[i])
        del all_flows[i]["occupancy"]
        del all_flows[i]["speed"]

    test = all_flows[0]
    for i in range(1, len(columns_flow), 1):
        test = test.append(all_flows[i])

    print('\nmerging flows...method 2...')
    test['flow_predicted'] = test.apply(lambda row: expected_flow(row), axis=1)
    global count
    count = 0
    test['confidence'] = test.apply(lambda row: expected_confidence(row), axis=1)

    for i in range(0, len(columns_flow), 1):
        del test["Expected_" + str(i+1)]
        del test["Confidence_" + str(i+1)]

    return test


def expected_flow(row):
    global count
    count += 1
    if count % 500000 == 0:
        print(str(count) + ' rows done..')

    expected = 0
    for i in range(0, len(flow_cols), 1):
        if not pd.isnull(row["Expected_" + str(i + 1)]):
            expected = row["Expected_" + str(i+1)]

    return expected


def expected_confidence(row):
    global count
    count += 1
    if count % 500000 == 0:
        print(str(count) + ' rows done..')

    confidence = 0
    for i in range(0, len(flow_cols), 1):
        if row["detector"] == "x" + str(i):
            confidence = row["Confidence_" + str(i+1)]

    return confidence


def predict_flow(timestamp, flow_x1, density, index):
    arr = []
    confidence = []

    arr.extend([flow_x1[0],flow_x1[1]])
    confidence.extend([density[0],density[1]])

    for i in range(2, len(timestamp)-2, 1):
        current_row_minus_two = timestamp[i-2]
        current_row_minus_one = timestamp[i-1]
        current_row = timestamp[i]
        current_row_plus_one = timestamp[i+1]
        current_row_plus_two = timestamp[i+2]
        
        time_difference = 150;
        
        kept_flow_preceding = []
        kept_density_preceding = []
        
        total_flow_preceding = 0
        total_density_preceding = 0
        
        kept_flow_following = []
        kept_density_following = []
        
        total_flow_following = 0
        total_density_following = 0

        bestConfidenceValue = 0
        
        W1 = 0
        W2 = 0
        
        if int((current_row - current_row_minus_two)/np.timedelta64(1,'s')) <= time_difference :
            kept_flow_preceding.append(flow_x1[i-2])
            kept_density_preceding.append(density[i-2])
            
        if int((current_row - current_row_minus_one)/np.timedelta64(1,'s')) <= time_difference :
            kept_flow_preceding.append(flow_x1[i-1])
            kept_density_preceding.append(density[i-1])
            
        if int((current_row_plus_one - current_row)/np.timedelta64(1,'s')) <= time_difference :
            kept_flow_following.append(flow_x1[i+1])
            kept_density_following.append(density[i+1])
            
        if int((current_row_plus_two - current_row)/np.timedelta64(1,'s')) <= time_difference :
            kept_flow_following.append(flow_x1[i+2])
            kept_density_following.append(density[i+2])
        
        for k in range(0,len(kept_flow_preceding),1):
            total_flow_preceding = total_flow_preceding + kept_flow_preceding[k]
            total_density_preceding = total_density_preceding + kept_density_preceding[k]
            
        for k in range(0,len(kept_flow_following),1):
            total_flow_following = total_flow_following + kept_flow_following[k]
            total_density_following = total_density_following + kept_density_following[k]
              
        if total_density_preceding != 0:
            W1 = total_density_preceding/(total_density_preceding + total_density_following)
        W2 = 1-W1
            
        if total_flow_preceding != 0 or total_flow_following != 0:
            average_flow = W1*(total_flow_preceding/2) + W2*(total_flow_following/2)
        else:
            average_flow = flow_x1[i]

        arr.append(average_flow)
        if len(kept_density_preceding) > 0:
            average_preceding = total_density_preceding / len(kept_density_preceding)
        else:
            average_preceding = 0
        if len(kept_density_following) > 0:
            average_following = total_density_following / len(kept_density_following)
        else:
            average_following = 0

        bestConfidenceValue = min(average_preceding, average_following)
        if not math.isnan(bestConfidenceValue) and bestConfidenceValue > 0:
            confidence.append(bestConfidenceValue)
        else:
            confidence.append(density[i])

        if i % 200000 == 0:
            print(i)
    
    arr.extend([flow_x1[len(flow_x1)-2],flow_x1[len(flow_x1)-1]])
    confidence.extend([density[len(density)-2],density[len(density)-1]])
    return arr,confidence
