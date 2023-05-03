USE alumini;
select Name from college_a_sj LIMIT 1000;
select * from college_a_se LIMIT 1000;
select * from college_a_hs LIMIT 1000;
select * from college_b_hs LIMIT 1000;
select * from college_b_se LIMIT 1000;
select * from college_b_sj LIMIT 1000;
SELECT * FROM college_a_se LIMIT 1500;
SELECT count(*) from C_A_SJ;
select * from c_a_sj LIMIT 1000;
select count(*) - count(null), count(null) from c_a_sj;
select count(null) from college_a_sj; -- no null values in any of the datasets.
CREATE VIEW C_A_SJ_1500 AS select * from college_a_sj LIMIT 1500;
select * from c_a_sj_1500;

CREATE VIEW C_A_HS_1500 AS select * from college_a_hs LIMIT 1500;
select * from c_a_hs_1500; -- only 1157 values present in this.

CREATE VIEW C_A_SE AS select * from college_a_se LIMIT 1000;
SELECT count(*) from C_A_SE;

CREATE VIEW C_B_HS AS select * from college_b_hs LIMIT 1000;
SELECT count(*) from C_B_HS;

CREATE VIEW C_B_SE AS select * from college_b_se LIMIT 1000;
SELECT count(*) from C_B_SE;

CREATE VIEW C_B_SJ AS select * from college_b_sj LIMIT 1000;
SELECT count(*) from C_B_SJ;










