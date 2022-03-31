FILE_NAME = 'data.csv'

OUTPUT_FILE_NAME = 'output.csv'

TARGET = 'ENERGY STAR Score'

CAT_FEATURES = [
                'Postal Code', 
                'Borough', 
                'Primary Property Type - Self Selected', 
                'Metered Areas (Energy)', 
                'Metered Areas  (Water)', 
                'Water Required?', 
                'NTA'
                ]

NUM_FEATURES = ['Largest Property Use Type - Gross Floor Area (ft²)',
                'Site EUI (kBtu/ft²)',
                'Weather Normalized Site EUI (kBtu/ft²)',
                'Weather Normalized Site Electricity Intensity (kWh/ft²)',
                'Weather Normalized Site Natural Gas Intensity (therms/ft²)',
                'Weather Normalized Source EUI (kBtu/ft²)',
                'Weather Normalized Site Natural Gas Use (therms)',
                'Electricity Use - Grid Purchase (kBtu)',
                'Weather Normalized Site Electricity (kWh)',
                'Total GHG Emissions (Metric Tons CO2e)',
                'Direct GHG Emissions (Metric Tons CO2e)',
                'Indirect GHG Emissions (Metric Tons CO2e)',
                'Source EUI (kBtu/ft²)',
                'DOF Gross Floor Area',
                'Year Built',
                'Number of Buildings - Self-reported',
                'Occupancy',
                'Property GFA - Self-Reported (ft²)',
                'Community Board',
                'Council District',
                'Census Tract']

MODEL_PARAMS = dict(    
                    thread_count=4,
                    depth=5,
                    n_estimators=1000,
                    learning_rate=0.1,
                    loss_function='RMSE',
                    verbose=50
)
