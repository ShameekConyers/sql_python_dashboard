# Per-Capita Normalization of Employment Series

Raw employment counts can be misleading when the working-age population changes over time. A growing population naturally increases the number of employees even if the employment rate is falling. Per-capita normalization removes this demographic effect.

## The Problem With Raw Counts

FRED series USINFO (information-sector employment) and CES2023800001 (semiconductor and electronic component manufacturing employment) report thousands of persons. Between 2016 and 2026 the U.S. civilian noninstitutional population aged 16 and over (FRED series CNP16OV) grew by roughly 10 million. A raw employment count that stayed flat during this period would actually represent declining employment intensity, because the same number of jobs now serves a larger population.

## CNP16OV as the Denominator

CNP16OV reports the total civilian noninstitutional population aged 16 and older, in thousands of persons, on a monthly basis. Dividing an employment count by CNP16OV and multiplying by 1,000 yields "employees per 1,000 working-age persons," an apples-to-apples metric that controls for population growth.

For example, if USINFO is 3,000 (thousand) and CNP16OV is 265,000 (thousand), the per-capita figure is 3000 / 265000 * 1000 = 11.32 information-sector employees per 1,000 working-age persons.

## Indexed Comparison

The dashboard indexes per-capita values to a common start date (the earliest observation for each series), setting the initial value to 100. This makes it easy to compare the trajectory of different series on the same chart even when their absolute per-capita levels differ by an order of magnitude.

## Impact on the Divergence Story

Without normalization, USINFO appears to have recovered to near its pre-pandemic peak in raw count terms. After normalization, the information sector's share of the working-age population has declined, reflecting AI-driven productivity gains and post-pandemic restructuring. CES2023800001 (semiconductor manufacturing) shows a sharper divergence: the CHIPS Act spurred capacity investment, but per-capita employment growth has been modest, suggesting automation absorbs much of the expansion.

## Series Not Requiring Normalization

UNRATE and U6RATE are already rates (percentages of the labor force). They do not need per-capita adjustment. CPIAUCSL is a price index, and T10Y2Y is a yield spread. Only absolute employment counts require this treatment.
