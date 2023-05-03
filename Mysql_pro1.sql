USE ecommerce;
SELECT * FROM users_data;
DESC users_data;
SELECT * FROM users_data LIMIT 100;
SELECT count(DISTINCT country) distinct_countries, count(DISTINCT language) distinct_languages FROM users_data;
SELECT gender,SUM(socialNbFollowers) followers from users_data group by gender;
SELECT count(*) hasprofilepic FROM users_data WHERE hasProfilePicture='True';
SELECT count(*) hasAnyApp FROM users_data WHERE hasAnyApp='True';
SELECT count(*) hasAndroidApp FROM users_data WHERE hasAndroidApp='True';
SELECT count(*) hasIosApp FROM users_data WHERE hasIosApp='True';

-- Calculate the total number of buyers for each country and sort the result in descending order of total number of buyers.
SELECT SUM(productsBought),country FROM users_data group by country ORDER BY SUM(productsBought)DESC;

-- Calculate the average number of sellers for each country and sort the result in ascending order of total number of sellers
SELECT country,AVG(productsSold) FROM users_data GROUP BY country ORDER BY AVG(productsSold)ASC;

-- Display name of top 10 countries having maximum products pass rate.
SELECT SUM(productsPassRate),country FROM users_data group by country ORDER BY SUM(productsPassRate)DESC LIMIT 10;

-- Calculate the number of users on an ecommerce platform for different language choices.
SELECT language,count(*) FROM users_data GROUP BY language;

-- Check the choice of female users about putting the product in a wishlist or to like socially on an ecommerce platform.
SELECT SUM(socialProductsLiked) liked,SUM(ProductsWished) wished FROM users_data where gender='F';

-- Check the choice of male users about being seller or buyer.
SELECT SUM(productsSold) seller,SUM(productsBought) buyer FROM users_data WHERE gender='M';

-- Which country is having maximum number of buyers?
SELECT country,SUM(productsBought) TOP_5 FROM users_data GROUP BY country ORDER BY SUM(productsBought) DESC LIMIT 5;

-- List the name of 10 countries having zero number of sellers.
SELECT country,productsSold FROM users_data WHERE productsSold=0 LIMIT 10;

-- Display record of top 110 users who have used ecommerce platform recently.
SELECT daysSinceLastLogin FROM users_data ORDER BY daysSinceLastLogin ASC LIMIT 110;

-- Calculate the number of female users those who have not logged in since last 100 days.
SELECT gender,COUNT(daysSinceLastLogin) FROM users_data WHERE gender='F' AND daysSinceLastLogin>100;

-- Display the number of female users of each country at ecommerce platform.
SELECT country,count(*) numberof_females FROM users_data WHERE gender='F' ORDER BY country;

-- Display the number of male users of each country at ecommerce platform.
SELECT country,count(*) numberof_males FROM users_data WHERE gender='M' ORDER BY country;

-- Calculate the average number of products sold and bought on ecommerce platform by male users for each country.
SELECT country,AVG(productsBought),AVG(productsSold) FROM users_data WHERE gender='M' GROUP BY country;

