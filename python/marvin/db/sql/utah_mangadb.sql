--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: functions; Type: SCHEMA; Schema: -; Owner: manga
--

CREATE SCHEMA functions;


ALTER SCHEMA functions OWNER TO manga;

--
-- Name: mangaauxdb; Type: SCHEMA; Schema: -; Owner: manga
--

CREATE SCHEMA mangaauxdb;


ALTER SCHEMA mangaauxdb OWNER TO manga;

--
-- Name: mangadatadb; Type: SCHEMA; Schema: -; Owner: manga
--

CREATE SCHEMA mangadatadb;


ALTER SCHEMA mangadatadb OWNER TO manga;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = functions, pg_catalog;

--
-- Name: q3c_ang2ipix(double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ang2ipix(double precision, double precision) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_ang2ipix';


ALTER FUNCTION functions.q3c_ang2ipix(double precision, double precision) OWNER TO manga;

--
-- Name: FUNCTION q3c_ang2ipix(double precision, double precision); Type: COMMENT; Schema: functions; Owner: manga
--

COMMENT ON FUNCTION q3c_ang2ipix(double precision, double precision) IS 'Function converting Ra and Dec to the Q3C ipix value';


--
-- Name: q3c_ang2ipix(real, real); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ang2ipix(ra real, decl real) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_ang2ipix_real';


ALTER FUNCTION functions.q3c_ang2ipix(ra real, decl real) OWNER TO manga;

--
-- Name: FUNCTION q3c_ang2ipix(ra real, decl real); Type: COMMENT; Schema: functions; Owner: manga
--

COMMENT ON FUNCTION q3c_ang2ipix(ra real, decl real) IS 'Function converting Ra and Dec(floats) to the Q3C ipix value';


--
-- Name: q3c_dist(double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_dist(ra1 double precision, dec1 double precision, ra2 double precision, dec2 double precision) RETURNS double precision
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_dist';


ALTER FUNCTION functions.q3c_dist(ra1 double precision, dec1 double precision, ra2 double precision, dec2 double precision) OWNER TO manga;

--
-- Name: FUNCTION q3c_dist(ra1 double precision, dec1 double precision, ra2 double precision, dec2 double precision); Type: COMMENT; Schema: functions; Owner: manga
--

COMMENT ON FUNCTION q3c_dist(ra1 double precision, dec1 double precision, ra2 double precision, dec2 double precision) IS 'Function q3c_dist(ra1, dec1, ra2, dec2) computing the distance between points (ra1, dec1) and (ra2, dec2)';


--
-- Name: q3c_ellipse_join(double precision, double precision, double precision, double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ellipse_join(leftra double precision, leftdec double precision, rightra double precision, rightdec double precision, majoraxis double precision, axisratio double precision, pa double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$
SELECT (((q3c_ang2ipix($3,$4)>=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,0))) AND (q3c_ang2ipix($3,$4)<=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,1))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,2))) AND (q3c_ang2ipix($3,$4)<=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,3))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,4))) AND (q3c_ang2ipix($3,$4)<=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,5))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,6))) AND (q3c_ang2ipix($3,$4)<=(q3c_ellipse_nearby_it($1,$2,$5,$6,$7,7))))) 
    AND q3c_in_ellipse($3,$4,$1,$2,$5,$6,$7)
$_$;


ALTER FUNCTION functions.q3c_ellipse_join(leftra double precision, leftdec double precision, rightra double precision, rightdec double precision, majoraxis double precision, axisratio double precision, pa double precision) OWNER TO manga;

--
-- Name: q3c_ellipse_nearby_it(double precision, double precision, double precision, double precision, double precision, integer); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ellipse_nearby_it(double precision, double precision, double precision, double precision, double precision, integer) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT COST 100
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_ellipse_nearby_it';


ALTER FUNCTION functions.q3c_ellipse_nearby_it(double precision, double precision, double precision, double precision, double precision, integer) OWNER TO manga;

--
-- Name: q3c_ellipse_query(double precision, double precision, double precision, double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ellipse_query(ra_col double precision, dec_col double precision, ra_ell double precision, dec_ell double precision, majax double precision, axis_ratio double precision, pa double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT (
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,0,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,1,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,2,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,3,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,4,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,5,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,6,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,7,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,8,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,9,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,10,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,11,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,12,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,13,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,14,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,15,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,16,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,17,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,18,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,19,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,20,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,21,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,22,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,23,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,24,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,25,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,26,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,27,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,28,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,29,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,30,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,31,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,32,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,33,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,34,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,35,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,36,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,37,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,38,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,39,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,40,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,41,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,42,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,43,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,44,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,45,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,46,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,47,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,48,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,49,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,50,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,51,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,52,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,53,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,54,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,55,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,56,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,57,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,58,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,59,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,60,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,61,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,62,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,63,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,64,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,65,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,66,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,67,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,68,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,69,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,70,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,71,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,72,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,73,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,74,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,75,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,76,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,77,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,78,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,79,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,80,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,81,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,82,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,83,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,84,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,85,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,86,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,87,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,88,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,89,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,90,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,91,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,92,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,93,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,94,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,95,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,96,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,97,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,98,1) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,99,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,0,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,1,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,2,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,3,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,4,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,5,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,6,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,7,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,8,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,9,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,10,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,11,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,12,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,13,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,14,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,15,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,16,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,17,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,18,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,19,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,20,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,21,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,22,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,23,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,24,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,25,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,26,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,27,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,28,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,29,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,30,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,31,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,32,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,33,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,34,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,35,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,36,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,37,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,38,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,39,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,40,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,41,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,42,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,43,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,44,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,45,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,46,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,47,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,48,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,49,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,50,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,51,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,52,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,53,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,54,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,55,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,56,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,57,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,58,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,59,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,60,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,61,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,62,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,63,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,64,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,65,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,66,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,67,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,68,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,69,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,70,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,71,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,72,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,73,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,74,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,75,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,76,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,77,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,78,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,79,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,80,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,81,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,82,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,83,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,84,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,85,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,86,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,87,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,88,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,89,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,90,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,91,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,92,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,93,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,94,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,95,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,96,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,97,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_ellipse_query_it($3,$4,$5,$6,$7,98,0) AND q3c_ang2ipix($1,$2)<q3c_ellipse_query_it($3,$4,$5,$6,$7,99,0)) 
) AND 
q3c_in_ellipse($1,$2,$3,$4,$5,$6,$7)
$_$;


ALTER FUNCTION functions.q3c_ellipse_query(ra_col double precision, dec_col double precision, ra_ell double precision, dec_ell double precision, majax double precision, axis_ratio double precision, pa double precision) OWNER TO manga;

--
-- Name: q3c_ellipse_query_it(double precision, double precision, double precision, double precision, double precision, integer, integer); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ellipse_query_it(ra_ell double precision, dec_ell double precision, majax double precision, axis_ratio double precision, pa double precision, iteration integer, full_flag integer) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_ellipse_query_it';


ALTER FUNCTION functions.q3c_ellipse_query_it(ra_ell double precision, dec_ell double precision, majax double precision, axis_ratio double precision, pa double precision, iteration integer, full_flag integer) OWNER TO manga;

--
-- Name: q3c_in_ellipse(double precision, double precision, double precision, double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_in_ellipse(ra0 double precision, dec0 double precision, ra_ell double precision, dec_ell double precision, maj_ax double precision, axis_ratio double precision, pa double precision) RETURNS boolean
    LANGUAGE c IMMUTABLE STRICT COST 100
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_in_ellipse';


ALTER FUNCTION functions.q3c_in_ellipse(ra0 double precision, dec0 double precision, ra_ell double precision, dec_ell double precision, maj_ax double precision, axis_ratio double precision, pa double precision) OWNER TO manga;

--
-- Name: q3c_in_poly(double precision, double precision, double precision[]); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_in_poly(double precision, double precision, double precision[]) RETURNS boolean
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_in_poly';


ALTER FUNCTION functions.q3c_in_poly(double precision, double precision, double precision[]) OWNER TO manga;

--
-- Name: q3c_ipix2ang(bigint); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ipix2ang(ipix bigint) RETURNS double precision[]
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_ipix2ang';


ALTER FUNCTION functions.q3c_ipix2ang(ipix bigint) OWNER TO manga;

--
-- Name: FUNCTION q3c_ipix2ang(ipix bigint); Type: COMMENT; Schema: functions; Owner: manga
--

COMMENT ON FUNCTION q3c_ipix2ang(ipix bigint) IS 'Function converting the Q3C ipix value to Ra, Dec';


--
-- Name: q3c_ipixcenter(double precision, double precision, integer); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_ipixcenter(ra double precision, decl double precision, integer) RETURNS bigint
    LANGUAGE sql
    AS $_$SELECT ((q3c_ang2ipix($1,$2))>>((2*$3))<<((2*$3))) +
			((1::bigint)<<(2*($3-1))) -1$_$;


ALTER FUNCTION functions.q3c_ipixcenter(ra double precision, decl double precision, integer) OWNER TO manga;

--
-- Name: q3c_join(double precision, double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_join(leftra double precision, leftdec double precision, rightra double precision, rightdec double precision, radius double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$
SELECT (((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,0))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,1))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,2))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,3))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,4))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,5))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,6))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,7))))) 
    AND q3c_sindist($1,$2,$3,$4)<POW(SIN(RADIANS($5)/2),2)
$_$;


ALTER FUNCTION functions.q3c_join(leftra double precision, leftdec double precision, rightra double precision, rightdec double precision, radius double precision) OWNER TO manga;

--
-- Name: q3c_join(double precision, double precision, real, real, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_join(leftra double precision, leftdec double precision, rightra real, rightdec real, radius double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$
SELECT (((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,0))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,1))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,2))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,3))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,4))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,5))))
    OR ((q3c_ang2ipix($3,$4)>=(q3c_nearby_it($1,$2,$5,6))) AND (q3c_ang2ipix($3,$4)<=(q3c_nearby_it($1,$2,$5,7))))) 
    AND q3c_sindist($1,$2,$3,$4)<POW(SIN(RADIANS($5)/2),2)
$_$;


ALTER FUNCTION functions.q3c_join(leftra double precision, leftdec double precision, rightra real, rightdec real, radius double precision) OWNER TO manga;

--
-- Name: q3c_join(double precision, double precision, double precision, double precision, bigint, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_join(double precision, double precision, double precision, double precision, bigint, double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT ((($5>=(q3c_nearby_it($1,$2,$6,0))) AND ($5<=(q3c_nearby_it($1,$2,$6,1))))
    OR (($5>=(q3c_nearby_it($1,$2,$6,2))) AND ($5<=(q3c_nearby_it($1,$2,$6,3))))
    OR (($5>=(q3c_nearby_it($1,$2,$6,4))) AND ($5<=(q3c_nearby_it($1,$2,$6,5))))
    OR (($5>=(q3c_nearby_it($1,$2,$6,6))) AND ($5<=(q3c_nearby_it($1,$2,$6,7))))) 
    AND q3c_sindist($1,$2,$3,$4)<POW(SIN(RADIANS($6)/2),2)
$_$;


ALTER FUNCTION functions.q3c_join(double precision, double precision, double precision, double precision, bigint, double precision) OWNER TO manga;

--
-- Name: q3c_nearby_it(double precision, double precision, double precision, integer); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_nearby_it(double precision, double precision, double precision, integer) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT COST 100
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_nearby_it';


ALTER FUNCTION functions.q3c_nearby_it(double precision, double precision, double precision, integer) OWNER TO manga;

--
-- Name: q3c_pixarea(bigint, integer); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_pixarea(ipix bigint, depth integer) RETURNS double precision
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_pixarea';


ALTER FUNCTION functions.q3c_pixarea(ipix bigint, depth integer) OWNER TO manga;

--
-- Name: FUNCTION q3c_pixarea(ipix bigint, depth integer); Type: COMMENT; Schema: functions; Owner: manga
--

COMMENT ON FUNCTION q3c_pixarea(ipix bigint, depth integer) IS 'Function returning the area of the pixel containing ipix being located at certain depth in the quadtree';


--
-- Name: q3c_poly_query(double precision, double precision, double precision[]); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_poly_query(double precision, double precision, double precision[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT 
(
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,0,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,1,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,2,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,3,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,4,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,5,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,6,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,7,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,8,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,9,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,10,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,11,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,12,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,13,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,14,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,15,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,16,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,17,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,18,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,19,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,20,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,21,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,22,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,23,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,24,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,25,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,26,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,27,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,28,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,29,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,30,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,31,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,32,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,33,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,34,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,35,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,36,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,37,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,38,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,39,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,40,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,41,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,42,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,43,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,44,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,45,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,46,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,47,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,48,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,49,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,50,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,51,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,52,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,53,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,54,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,55,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,56,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,57,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,58,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,59,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,60,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,61,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,62,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,63,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,64,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,65,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,66,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,67,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,68,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,69,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,70,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,71,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,72,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,73,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,74,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,75,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,76,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,77,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,78,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,79,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,80,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,81,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,82,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,83,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,84,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,85,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,86,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,87,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,88,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,89,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,90,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,91,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,92,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,93,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,94,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,95,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,96,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,97,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,98,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,99,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,0,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,1,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,2,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,3,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,4,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,5,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,6,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,7,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,8,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,9,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,10,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,11,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,12,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,13,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,14,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,15,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,16,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,17,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,18,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,19,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,20,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,21,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,22,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,23,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,24,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,25,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,26,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,27,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,28,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,29,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,30,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,31,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,32,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,33,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,34,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,35,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,36,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,37,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,38,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,39,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,40,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,41,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,42,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,43,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,44,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,45,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,46,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,47,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,48,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,49,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,50,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,51,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,52,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,53,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,54,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,55,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,56,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,57,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,58,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,59,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,60,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,61,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,62,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,63,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,64,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,65,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,66,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,67,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,68,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,69,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,70,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,71,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,72,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,73,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,74,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,75,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,76,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,77,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,78,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,79,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,80,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,81,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,82,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,83,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,84,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,85,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,86,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,87,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,88,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,89,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,90,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,91,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,92,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,93,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,94,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,95,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,96,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,97,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,98,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,99,0)) 
) AND 
q3c_in_poly($1,$2,$3);
$_$;


ALTER FUNCTION functions.q3c_poly_query(double precision, double precision, double precision[]) OWNER TO manga;

--
-- Name: q3c_poly_query(real, real, double precision[]); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_poly_query(real, real, double precision[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT 
(
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,0,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,1,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,2,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,3,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,4,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,5,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,6,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,7,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,8,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,9,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,10,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,11,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,12,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,13,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,14,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,15,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,16,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,17,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,18,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,19,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,20,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,21,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,22,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,23,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,24,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,25,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,26,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,27,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,28,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,29,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,30,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,31,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,32,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,33,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,34,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,35,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,36,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,37,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,38,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,39,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,40,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,41,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,42,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,43,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,44,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,45,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,46,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,47,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,48,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,49,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,50,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,51,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,52,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,53,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,54,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,55,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,56,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,57,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,58,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,59,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,60,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,61,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,62,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,63,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,64,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,65,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,66,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,67,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,68,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,69,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,70,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,71,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,72,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,73,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,74,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,75,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,76,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,77,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,78,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,79,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,80,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,81,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,82,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,83,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,84,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,85,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,86,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,87,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,88,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,89,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,90,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,91,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,92,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,93,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,94,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,95,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,96,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,97,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,98,1) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,99,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,0,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,1,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,2,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,3,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,4,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,5,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,6,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,7,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,8,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,9,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,10,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,11,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,12,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,13,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,14,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,15,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,16,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,17,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,18,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,19,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,20,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,21,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,22,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,23,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,24,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,25,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,26,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,27,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,28,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,29,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,30,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,31,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,32,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,33,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,34,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,35,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,36,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,37,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,38,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,39,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,40,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,41,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,42,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,43,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,44,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,45,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,46,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,47,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,48,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,49,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,50,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,51,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,52,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,53,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,54,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,55,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,56,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,57,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,58,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,59,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,60,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,61,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,62,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,63,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,64,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,65,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,66,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,67,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,68,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,69,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,70,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,71,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,72,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,73,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,74,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,75,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,76,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,77,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,78,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,79,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,80,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,81,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,82,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,83,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,84,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,85,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,86,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,87,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,88,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,89,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,90,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,91,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,92,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,93,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,94,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,95,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,96,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,97,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_poly_query_it($3,98,0) AND q3c_ang2ipix($1,$2)<q3c_poly_query_it($3,99,0)) 
) AND 
q3c_in_poly($1,$2,$3) ;
$_$;


ALTER FUNCTION functions.q3c_poly_query(real, real, double precision[]) OWNER TO manga;

--
-- Name: q3c_poly_query_it(double precision[], integer, integer); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_poly_query_it(double precision[], integer, integer) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_poly_query_it';


ALTER FUNCTION functions.q3c_poly_query_it(double precision[], integer, integer) OWNER TO manga;

--
-- Name: q3c_radial_query(double precision, double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_radial_query(double precision, double precision, double precision, double precision, double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT 
(
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,0,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,1,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,2,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,3,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,4,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,5,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,6,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,7,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,8,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,9,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,10,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,11,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,12,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,13,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,14,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,15,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,16,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,17,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,18,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,19,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,20,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,21,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,22,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,23,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,24,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,25,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,26,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,27,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,28,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,29,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,30,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,31,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,32,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,33,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,34,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,35,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,36,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,37,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,38,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,39,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,40,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,41,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,42,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,43,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,44,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,45,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,46,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,47,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,48,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,49,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,50,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,51,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,52,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,53,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,54,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,55,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,56,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,57,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,58,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,59,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,60,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,61,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,62,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,63,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,64,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,65,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,66,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,67,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,68,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,69,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,70,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,71,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,72,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,73,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,74,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,75,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,76,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,77,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,78,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,79,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,80,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,81,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,82,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,83,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,84,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,85,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,86,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,87,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,88,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,89,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,90,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,91,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,92,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,93,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,94,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,95,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,96,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,97,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,98,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,99,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,0,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,1,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,2,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,3,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,4,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,5,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,6,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,7,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,8,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,9,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,10,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,11,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,12,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,13,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,14,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,15,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,16,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,17,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,18,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,19,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,20,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,21,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,22,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,23,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,24,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,25,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,26,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,27,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,28,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,29,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,30,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,31,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,32,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,33,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,34,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,35,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,36,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,37,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,38,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,39,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,40,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,41,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,42,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,43,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,44,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,45,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,46,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,47,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,48,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,49,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,50,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,51,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,52,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,53,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,54,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,55,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,56,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,57,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,58,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,59,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,60,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,61,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,62,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,63,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,64,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,65,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,66,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,67,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,68,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,69,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,70,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,71,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,72,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,73,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,74,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,75,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,76,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,77,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,78,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,79,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,80,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,81,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,82,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,83,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,84,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,85,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,86,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,87,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,88,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,89,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,90,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,91,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,92,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,93,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,94,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,95,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,96,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,97,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,98,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,99,0)) 
) AND
q3c_sindist($1,$2,$3,$4)<POW(SIN(RADIANS($5)/2),2)
$_$;


ALTER FUNCTION functions.q3c_radial_query(double precision, double precision, double precision, double precision, double precision) OWNER TO manga;

--
-- Name: q3c_radial_query(real, real, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_radial_query(real, real, double precision, double precision, double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT (
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,0,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,1,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,2,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,3,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,4,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,5,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,6,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,7,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,8,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,9,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,10,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,11,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,12,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,13,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,14,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,15,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,16,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,17,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,18,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,19,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,20,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,21,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,22,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,23,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,24,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,25,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,26,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,27,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,28,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,29,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,30,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,31,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,32,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,33,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,34,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,35,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,36,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,37,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,38,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,39,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,40,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,41,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,42,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,43,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,44,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,45,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,46,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,47,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,48,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,49,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,50,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,51,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,52,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,53,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,54,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,55,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,56,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,57,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,58,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,59,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,60,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,61,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,62,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,63,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,64,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,65,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,66,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,67,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,68,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,69,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,70,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,71,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,72,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,73,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,74,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,75,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,76,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,77,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,78,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,79,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,80,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,81,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,82,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,83,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,84,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,85,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,86,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,87,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,88,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,89,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,90,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,91,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,92,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,93,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,94,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,95,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,96,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,97,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,98,1) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,99,1)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,0,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,1,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,2,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,3,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,4,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,5,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,6,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,7,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,8,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,9,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,10,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,11,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,12,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,13,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,14,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,15,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,16,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,17,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,18,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,19,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,20,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,21,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,22,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,23,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,24,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,25,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,26,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,27,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,28,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,29,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,30,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,31,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,32,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,33,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,34,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,35,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,36,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,37,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,38,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,39,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,40,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,41,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,42,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,43,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,44,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,45,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,46,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,47,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,48,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,49,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,50,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,51,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,52,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,53,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,54,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,55,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,56,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,57,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,58,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,59,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,60,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,61,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,62,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,63,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,64,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,65,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,66,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,67,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,68,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,69,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,70,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,71,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,72,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,73,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,74,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,75,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,76,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,77,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,78,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,79,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,80,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,81,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,82,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,83,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,84,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,85,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,86,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,87,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,88,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,89,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,90,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,91,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,92,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,93,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,94,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,95,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,96,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,97,0)) OR
(q3c_ang2ipix($1,$2)>=q3c_radial_query_it($3,$4,$5,98,0) AND q3c_ang2ipix($1,$2)<q3c_radial_query_it($3,$4,$5,99,0)) 
) AND
q3c_sindist($1,$2,$3,$4)<POW(SIN(RADIANS($5)/2),2)
$_$;


ALTER FUNCTION functions.q3c_radial_query(real, real, double precision, double precision, double precision) OWNER TO manga;

--
-- Name: q3c_radial_query(bigint, double precision, double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_radial_query(bigint, double precision, double precision, double precision, double precision, double precision) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT 
(
($1>=q3c_radial_query_it($4,$5,$6,0,1) AND $1<q3c_radial_query_it($4,$5,$6,1,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,2,1) AND $1<q3c_radial_query_it($4,$5,$6,3,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,4,1) AND $1<q3c_radial_query_it($4,$5,$6,5,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,6,1) AND $1<q3c_radial_query_it($4,$5,$6,7,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,8,1) AND $1<q3c_radial_query_it($4,$5,$6,9,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,10,1) AND $1<q3c_radial_query_it($4,$5,$6,11,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,12,1) AND $1<q3c_radial_query_it($4,$5,$6,13,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,14,1) AND $1<q3c_radial_query_it($4,$5,$6,15,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,16,1) AND $1<q3c_radial_query_it($4,$5,$6,17,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,18,1) AND $1<q3c_radial_query_it($4,$5,$6,19,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,20,1) AND $1<q3c_radial_query_it($4,$5,$6,21,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,22,1) AND $1<q3c_radial_query_it($4,$5,$6,23,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,24,1) AND $1<q3c_radial_query_it($4,$5,$6,25,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,26,1) AND $1<q3c_radial_query_it($4,$5,$6,27,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,28,1) AND $1<q3c_radial_query_it($4,$5,$6,29,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,30,1) AND $1<q3c_radial_query_it($4,$5,$6,31,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,32,1) AND $1<q3c_radial_query_it($4,$5,$6,33,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,34,1) AND $1<q3c_radial_query_it($4,$5,$6,35,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,36,1) AND $1<q3c_radial_query_it($4,$5,$6,37,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,38,1) AND $1<q3c_radial_query_it($4,$5,$6,39,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,40,1) AND $1<q3c_radial_query_it($4,$5,$6,41,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,42,1) AND $1<q3c_radial_query_it($4,$5,$6,43,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,44,1) AND $1<q3c_radial_query_it($4,$5,$6,45,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,46,1) AND $1<q3c_radial_query_it($4,$5,$6,47,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,48,1) AND $1<q3c_radial_query_it($4,$5,$6,49,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,50,1) AND $1<q3c_radial_query_it($4,$5,$6,51,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,52,1) AND $1<q3c_radial_query_it($4,$5,$6,53,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,54,1) AND $1<q3c_radial_query_it($4,$5,$6,55,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,56,1) AND $1<q3c_radial_query_it($4,$5,$6,57,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,58,1) AND $1<q3c_radial_query_it($4,$5,$6,59,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,60,1) AND $1<q3c_radial_query_it($4,$5,$6,61,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,62,1) AND $1<q3c_radial_query_it($4,$5,$6,63,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,64,1) AND $1<q3c_radial_query_it($4,$5,$6,65,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,66,1) AND $1<q3c_radial_query_it($4,$5,$6,67,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,68,1) AND $1<q3c_radial_query_it($4,$5,$6,69,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,70,1) AND $1<q3c_radial_query_it($4,$5,$6,71,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,72,1) AND $1<q3c_radial_query_it($4,$5,$6,73,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,74,1) AND $1<q3c_radial_query_it($4,$5,$6,75,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,76,1) AND $1<q3c_radial_query_it($4,$5,$6,77,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,78,1) AND $1<q3c_radial_query_it($4,$5,$6,79,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,80,1) AND $1<q3c_radial_query_it($4,$5,$6,81,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,82,1) AND $1<q3c_radial_query_it($4,$5,$6,83,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,84,1) AND $1<q3c_radial_query_it($4,$5,$6,85,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,86,1) AND $1<q3c_radial_query_it($4,$5,$6,87,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,88,1) AND $1<q3c_radial_query_it($4,$5,$6,89,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,90,1) AND $1<q3c_radial_query_it($4,$5,$6,91,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,92,1) AND $1<q3c_radial_query_it($4,$5,$6,93,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,94,1) AND $1<q3c_radial_query_it($4,$5,$6,95,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,96,1) AND $1<q3c_radial_query_it($4,$5,$6,97,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,98,1) AND $1<q3c_radial_query_it($4,$5,$6,99,1)) OR
($1>=q3c_radial_query_it($4,$5,$6,0,0) AND $1<q3c_radial_query_it($4,$5,$6,1,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,2,0) AND $1<q3c_radial_query_it($4,$5,$6,3,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,4,0) AND $1<q3c_radial_query_it($4,$5,$6,5,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,6,0) AND $1<q3c_radial_query_it($4,$5,$6,7,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,8,0) AND $1<q3c_radial_query_it($4,$5,$6,9,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,10,0) AND $1<q3c_radial_query_it($4,$5,$6,11,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,12,0) AND $1<q3c_radial_query_it($4,$5,$6,13,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,14,0) AND $1<q3c_radial_query_it($4,$5,$6,15,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,16,0) AND $1<q3c_radial_query_it($4,$5,$6,17,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,18,0) AND $1<q3c_radial_query_it($4,$5,$6,19,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,20,0) AND $1<q3c_radial_query_it($4,$5,$6,21,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,22,0) AND $1<q3c_radial_query_it($4,$5,$6,23,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,24,0) AND $1<q3c_radial_query_it($4,$5,$6,25,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,26,0) AND $1<q3c_radial_query_it($4,$5,$6,27,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,28,0) AND $1<q3c_radial_query_it($4,$5,$6,29,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,30,0) AND $1<q3c_radial_query_it($4,$5,$6,31,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,32,0) AND $1<q3c_radial_query_it($4,$5,$6,33,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,34,0) AND $1<q3c_radial_query_it($4,$5,$6,35,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,36,0) AND $1<q3c_radial_query_it($4,$5,$6,37,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,38,0) AND $1<q3c_radial_query_it($4,$5,$6,39,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,40,0) AND $1<q3c_radial_query_it($4,$5,$6,41,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,42,0) AND $1<q3c_radial_query_it($4,$5,$6,43,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,44,0) AND $1<q3c_radial_query_it($4,$5,$6,45,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,46,0) AND $1<q3c_radial_query_it($4,$5,$6,47,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,48,0) AND $1<q3c_radial_query_it($4,$5,$6,49,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,50,0) AND $1<q3c_radial_query_it($4,$5,$6,51,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,52,0) AND $1<q3c_radial_query_it($4,$5,$6,53,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,54,0) AND $1<q3c_radial_query_it($4,$5,$6,55,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,56,0) AND $1<q3c_radial_query_it($4,$5,$6,57,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,58,0) AND $1<q3c_radial_query_it($4,$5,$6,59,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,60,0) AND $1<q3c_radial_query_it($4,$5,$6,61,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,62,0) AND $1<q3c_radial_query_it($4,$5,$6,63,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,64,0) AND $1<q3c_radial_query_it($4,$5,$6,65,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,66,0) AND $1<q3c_radial_query_it($4,$5,$6,67,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,68,0) AND $1<q3c_radial_query_it($4,$5,$6,69,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,70,0) AND $1<q3c_radial_query_it($4,$5,$6,71,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,72,0) AND $1<q3c_radial_query_it($4,$5,$6,73,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,74,0) AND $1<q3c_radial_query_it($4,$5,$6,75,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,76,0) AND $1<q3c_radial_query_it($4,$5,$6,77,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,78,0) AND $1<q3c_radial_query_it($4,$5,$6,79,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,80,0) AND $1<q3c_radial_query_it($4,$5,$6,81,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,82,0) AND $1<q3c_radial_query_it($4,$5,$6,83,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,84,0) AND $1<q3c_radial_query_it($4,$5,$6,85,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,86,0) AND $1<q3c_radial_query_it($4,$5,$6,87,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,88,0) AND $1<q3c_radial_query_it($4,$5,$6,89,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,90,0) AND $1<q3c_radial_query_it($4,$5,$6,91,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,92,0) AND $1<q3c_radial_query_it($4,$5,$6,93,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,94,0) AND $1<q3c_radial_query_it($4,$5,$6,95,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,96,0) AND $1<q3c_radial_query_it($4,$5,$6,97,0)) OR
($1>=q3c_radial_query_it($4,$5,$6,98,0) AND $1<q3c_radial_query_it($4,$5,$6,99,0)) 
) AND 
q3c_sindist($2,$3,$4,$5)<POW(SIN(RADIANS($6)/2),2)

$_$;


ALTER FUNCTION functions.q3c_radial_query(bigint, double precision, double precision, double precision, double precision, double precision) OWNER TO manga;

--
-- Name: q3c_radial_query_it(double precision, double precision, double precision, integer, integer); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_radial_query_it(double precision, double precision, double precision, integer, integer) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_radial_query_it';


ALTER FUNCTION functions.q3c_radial_query_it(double precision, double precision, double precision, integer, integer) OWNER TO manga;

--
-- Name: q3c_sindist(double precision, double precision, double precision, double precision); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_sindist(double precision, double precision, double precision, double precision) RETURNS double precision
    LANGUAGE c IMMUTABLE STRICT COST 100
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_sindist';


ALTER FUNCTION functions.q3c_sindist(double precision, double precision, double precision, double precision) OWNER TO manga;

--
-- Name: q3c_version(); Type: FUNCTION; Schema: functions; Owner: manga
--

CREATE FUNCTION q3c_version() RETURNS cstring
    LANGUAGE c IMMUTABLE STRICT
    AS '/uufs/chpc.utah.edu/sys/pkg/q3c/1.4.23/q3c', 'pgq3c_get_version';


ALTER FUNCTION functions.q3c_version() OWNER TO manga;

--
-- Name: FUNCTION q3c_version(); Type: COMMENT; Schema: functions; Owner: manga
--

COMMENT ON FUNCTION q3c_version() IS 'Function returning Q3C version';


SET search_path = mangaauxdb, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: cube_header; Type: TABLE; Schema: mangaauxdb; Owner: manga; Tablespace: 
--

CREATE TABLE cube_header (
    pk integer NOT NULL,
    header json,
    cube_pk integer
);


ALTER TABLE mangaauxdb.cube_header OWNER TO manga;

--
-- Name: cube_header_pk_seq; Type: SEQUENCE; Schema: mangaauxdb; Owner: manga
--

CREATE SEQUENCE cube_header_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangaauxdb.cube_header_pk_seq OWNER TO manga;

--
-- Name: cube_header_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangaauxdb; Owner: manga
--

ALTER SEQUENCE cube_header_pk_seq OWNED BY cube_header.pk;


--
-- Name: maskbit; Type: TABLE; Schema: mangaauxdb; Owner: manga; Tablespace: 
--

CREATE TABLE maskbit (
    pk integer NOT NULL,
    flag text,
    "bit" integer,
    label text,
    description text
);


ALTER TABLE mangaauxdb.maskbit OWNER TO manga;

--
-- Name: maskbit_labels; Type: TABLE; Schema: mangaauxdb; Owner: manga; Tablespace: 
--

CREATE TABLE maskbit_labels (
    pk integer NOT NULL,
    maskbit integer,
    labels json,
    flag text
);


ALTER TABLE mangaauxdb.maskbit_labels OWNER TO manga;

--
-- Name: maskbit_labels_pk_seq; Type: SEQUENCE; Schema: mangaauxdb; Owner: manga
--

CREATE SEQUENCE maskbit_labels_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangaauxdb.maskbit_labels_pk_seq OWNER TO manga;

--
-- Name: maskbit_labels_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangaauxdb; Owner: manga
--

ALTER SEQUENCE maskbit_labels_pk_seq OWNED BY maskbit_labels.pk;


--
-- Name: maskbit_pk_seq; Type: SEQUENCE; Schema: mangaauxdb; Owner: manga
--

CREATE SEQUENCE maskbit_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangaauxdb.maskbit_pk_seq OWNER TO manga;

--
-- Name: maskbit_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangaauxdb; Owner: manga
--

ALTER SEQUENCE maskbit_pk_seq OWNED BY maskbit.pk;


SET search_path = mangadatadb, pg_catalog;

--
-- Name: cart; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE cart (
    pk integer NOT NULL,
    id integer
);


ALTER TABLE mangadatadb.cart OWNER TO manga;

--
-- Name: cart_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE cart_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.cart_pk_seq OWNER TO manga;

--
-- Name: cart_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE cart_pk_seq OWNED BY cart.pk;


--
-- Name: cart_to_cube; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE cart_to_cube (
    pk integer NOT NULL,
    cube_pk integer,
    cart_pk integer
);


ALTER TABLE mangadatadb.cart_to_cube OWNER TO manga;

--
-- Name: cart_to_cube_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE cart_to_cube_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.cart_to_cube_pk_seq OWNER TO manga;

--
-- Name: cart_to_cube_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE cart_to_cube_pk_seq OWNED BY cart_to_cube.pk;


--
-- Name: cube; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE cube (
    pk integer NOT NULL,
    plate integer,
    mangaid text,
    designid integer,
    pipeline_info_pk integer,
    wavelength_pk integer,
    ifudesign_pk integer,
    specres real[],
    xfocal real,
    yfocal real,
    ra double precision,
    "dec" double precision
);


ALTER TABLE mangadatadb.cube OWNER TO manga;

--
-- Name: cube_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE cube_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.cube_pk_seq OWNER TO manga;

--
-- Name: cube_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE cube_pk_seq OWNED BY cube.pk;


--
-- Name: fiber_type; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE fiber_type (
    pk integer NOT NULL,
    label text
);


ALTER TABLE mangadatadb.fiber_type OWNER TO manga;

--
-- Name: fiber_type_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE fiber_type_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.fiber_type_pk_seq OWNER TO manga;

--
-- Name: fiber_type_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE fiber_type_pk_seq OWNED BY fiber_type.pk;


--
-- Name: fibers; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE fibers (
    pk integer NOT NULL,
    fiberid integer,
    specfibid integer,
    fnum integer,
    ring integer,
    dist_mm real,
    xpmm real,
    ypmm real,
    fiber_type_pk integer,
    target_type_pk integer,
    ifudesign_pk integer
);


ALTER TABLE mangadatadb.fibers OWNER TO manga;

--
-- Name: fibers_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE fibers_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.fibers_pk_seq OWNER TO manga;

--
-- Name: fibers_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE fibers_pk_seq OWNED BY fibers.pk;


--
-- Name: fits_header_keyword; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE fits_header_keyword (
    pk integer NOT NULL,
    label text
);


ALTER TABLE mangadatadb.fits_header_keyword OWNER TO manga;

--
-- Name: fits_header_keyword_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE fits_header_keyword_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.fits_header_keyword_pk_seq OWNER TO manga;

--
-- Name: fits_header_keyword_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE fits_header_keyword_pk_seq OWNED BY fits_header_keyword.pk;


--
-- Name: fits_header_value; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE fits_header_value (
    pk integer NOT NULL,
    value text,
    index integer,
    comment text,
    fits_header_keyword_pk integer,
    cube_pk integer
);


ALTER TABLE mangadatadb.fits_header_value OWNER TO manga;

--
-- Name: fits_header_value_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE fits_header_value_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.fits_header_value_pk_seq OWNER TO manga;

--
-- Name: fits_header_value_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE fits_header_value_pk_seq OWNED BY fits_header_value.pk;


--
-- Name: ifu_to_block; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE ifu_to_block (
    pk integer NOT NULL,
    ifudesign_pk integer,
    slitblock_pk integer
);


ALTER TABLE mangadatadb.ifu_to_block OWNER TO manga;

--
-- Name: ifu_to_block_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE ifu_to_block_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.ifu_to_block_pk_seq OWNER TO manga;

--
-- Name: ifu_to_block_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE ifu_to_block_pk_seq OWNED BY ifu_to_block.pk;


--
-- Name: ifudesign; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE ifudesign (
    pk integer NOT NULL,
    name text,
    nfiber integer,
    nsky integer,
    nblocks integer,
    specid integer,
    maxring integer
);


ALTER TABLE mangadatadb.ifudesign OWNER TO manga;

--
-- Name: ifudesign_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE ifudesign_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.ifudesign_pk_seq OWNER TO manga;

--
-- Name: ifudesign_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE ifudesign_pk_seq OWNED BY ifudesign.pk;


--
-- Name: pipeline_completion_status; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE pipeline_completion_status (
    pk integer NOT NULL,
    label text
);


ALTER TABLE mangadatadb.pipeline_completion_status OWNER TO manga;

--
-- Name: pipeline_completion_status_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE pipeline_completion_status_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.pipeline_completion_status_pk_seq OWNER TO manga;

--
-- Name: pipeline_completion_status_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE pipeline_completion_status_pk_seq OWNED BY pipeline_completion_status.pk;


--
-- Name: pipeline_info; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE pipeline_info (
    pk integer NOT NULL,
    pipeline_name_pk integer,
    pipeline_stage_pk integer,
    pipeline_version_pk integer,
    pipeline_completion_status_pk integer
);


ALTER TABLE mangadatadb.pipeline_info OWNER TO manga;

--
-- Name: pipeline_info_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE pipeline_info_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.pipeline_info_pk_seq OWNER TO manga;

--
-- Name: pipeline_info_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE pipeline_info_pk_seq OWNED BY pipeline_info.pk;


--
-- Name: pipeline_name; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE pipeline_name (
    pk integer NOT NULL,
    label text
);


ALTER TABLE mangadatadb.pipeline_name OWNER TO manga;

--
-- Name: pipeline_name_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE pipeline_name_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.pipeline_name_pk_seq OWNER TO manga;

--
-- Name: pipeline_name_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE pipeline_name_pk_seq OWNED BY pipeline_name.pk;


--
-- Name: pipeline_stage; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE pipeline_stage (
    pk integer NOT NULL,
    label text
);


ALTER TABLE mangadatadb.pipeline_stage OWNER TO manga;

--
-- Name: pipeline_stage_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE pipeline_stage_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.pipeline_stage_pk_seq OWNER TO manga;

--
-- Name: pipeline_stage_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE pipeline_stage_pk_seq OWNED BY pipeline_stage.pk;


--
-- Name: pipeline_version; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE pipeline_version (
    pk integer NOT NULL,
    version text
);


ALTER TABLE mangadatadb.pipeline_version OWNER TO manga;

--
-- Name: pipeline_version_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE pipeline_version_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.pipeline_version_pk_seq OWNER TO manga;

--
-- Name: pipeline_version_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE pipeline_version_pk_seq OWNED BY pipeline_version.pk;


--
-- Name: rssfiber; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE rssfiber (
    pk integer NOT NULL,
    flux real[],
    ivar real[],
    mask integer[],
    xpos real[],
    ypos real[],
    exposure_no integer,
    mjd integer,
    exposure_pk integer,
    cube_pk integer
);


ALTER TABLE mangadatadb.rssfiber OWNER TO manga;

--
-- Name: rssfiber_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE rssfiber_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.rssfiber_pk_seq OWNER TO manga;

--
-- Name: rssfiber_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE rssfiber_pk_seq OWNED BY rssfiber.pk;


--
-- Name: sample; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE sample (
    pk integer NOT NULL,
    manga_tileid integer,
    ifu_ra double precision,
    ifu_dec double precision,
    target_ra double precision,
    target_dec double precision,
    iauname text,
    ifudesignsize integer,
    ifutargetsize integer,
    ifudesignwrongsize integer,
    field integer,
    run integer,
    nsa_redshift real,
    nsa_zdist real,
    nsa_absmag_f real,
    nsa_absmag_n real,
    nsa_absmag_u real,
    nsa_absmag_g real,
    nsa_absmag_r real,
    nsa_absmag_i real,
    nsa_absmag_z real,
    nsa_mstar real,
    nsa_vdisp real,
    nsa_inclination real,
    nsa_petro_th50 real,
    nsa_petroflux_f real,
    nsa_petroflux_n real,
    nsa_petroflux_u real,
    nsa_petroflux_g real,
    nsa_petroflux_r real,
    nsa_petroflux_i real,
    nsa_petroflux_z real,
    nsa_petroflux_ivar_f real,
    nsa_petroflux_ivar_n real,
    nsa_petroflux_ivar_u real,
    nsa_petroflux_ivar_g real,
    nsa_petroflux_ivar_r real,
    nsa_petroflux_ivar_i real,
    nsa_petroflux_ivar_z real,
    nsa_sersic_ba real,
    nsa_sersic_n real,
    nsa_sersic_phi real,
    nsa_sersic_th50 real,
    nsa_sersicflux_f real,
    nsa_sersicflux_n real,
    nsa_sersicflux_u real,
    nsa_sersicflux_g real,
    nsa_sersicflux_r real,
    nsa_sersicflux_i real,
    nsa_sersicflux_z real,
    nsa_sersicflux_ivar_f real,
    nsa_sersicflux_ivar_n real,
    nsa_sersicflux_ivar_u real,
    nsa_sersicflux_ivar_g real,
    nsa_sersicflux_ivar_r real,
    nsa_sersicflux_ivar_i real,
    nsa_sersicflux_ivar_z real,
    cube_pk integer,
    nsa_version text,
    nsa_id bigint,
    nsa_id100 bigint,
    nsa_ba real,
    nsa_phi real,
    nsa_mstar_el real,
    nsa_petro_th50_el real,
    nsa_extinction_f real,
    nsa_extinction_n real,
    nsa_extinction_u real,
    nsa_extinction_g real,
    nsa_extinction_r real,
    nsa_extinction_i real,
    nsa_extinction_z real,
    nsa_amivar_el_f real,
    nsa_amivar_el_n real,
    nsa_amivar_el_u real,
    nsa_amivar_el_g real,
    nsa_amivar_el_r real,
    nsa_amivar_el_i real,
    nsa_amivar_el_z real,
    nsa_petroflux_el_f real,
    nsa_petroflux_el_n real,
    nsa_petroflux_el_u real,
    nsa_petroflux_el_g real,
    nsa_petroflux_el_r real,
    nsa_petroflux_el_i real,
    nsa_petroflux_el_z real,
    nsa_petroflux_el_ivar_f real,
    nsa_petroflux_el_ivar_n real,
    nsa_petroflux_el_ivar_u real,
    nsa_petroflux_el_ivar_g real,
    nsa_petroflux_el_ivar_r real,
    nsa_petroflux_el_ivar_i real,
    nsa_petroflux_el_ivar_z real,
    nsa_absmag_el_f real,
    nsa_absmag_el_n real,
    nsa_absmag_el_u real,
    nsa_absmag_el_g real,
    nsa_absmag_el_r real,
    nsa_absmag_el_i real,
    nsa_absmag_el_z real
);


ALTER TABLE mangadatadb.sample OWNER TO manga;

--
-- Name: sample_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE sample_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.sample_pk_seq OWNER TO manga;

--
-- Name: sample_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE sample_pk_seq OWNED BY sample.pk;


--
-- Name: slitblock; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE slitblock (
    pk integer NOT NULL,
    blockid integer,
    specblockid integer,
    nfiber integer
);


ALTER TABLE mangadatadb.slitblock OWNER TO manga;

--
-- Name: slitblock_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE slitblock_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.slitblock_pk_seq OWNER TO manga;

--
-- Name: slitblock_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE slitblock_pk_seq OWNED BY slitblock.pk;


--
-- Name: spaxel; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE spaxel (
    pk integer NOT NULL,
    flux real[],
    ivar real[],
    mask integer[],
    cube_pk integer
);


ALTER TABLE mangadatadb.spaxel OWNER TO manga;

--
-- Name: spaxel_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE spaxel_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.spaxel_pk_seq OWNER TO manga;

--
-- Name: spaxel_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE spaxel_pk_seq OWNED BY spaxel.pk;


--
-- Name: target_type; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE target_type (
    pk integer NOT NULL,
    label text
);


ALTER TABLE mangadatadb.target_type OWNER TO manga;

--
-- Name: target_type_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE target_type_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.target_type_pk_seq OWNER TO manga;

--
-- Name: target_type_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE target_type_pk_seq OWNED BY target_type.pk;


--
-- Name: test_rssfiber; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE test_rssfiber (
    pk integer NOT NULL,
    flux json,
    cube_pk integer
);


ALTER TABLE mangadatadb.test_rssfiber OWNER TO manga;

--
-- Name: test_rssfiber_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE test_rssfiber_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.test_rssfiber_pk_seq OWNER TO manga;

--
-- Name: test_rssfiber_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE test_rssfiber_pk_seq OWNED BY test_rssfiber.pk;


--
-- Name: test_spaxel; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE test_spaxel (
    pk integer NOT NULL,
    flux real[],
    ivar real[],
    mask integer[],
    cube_pk integer,
    flux_json json
);


ALTER TABLE mangadatadb.test_spaxel OWNER TO manga;

--
-- Name: test_spaxel_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE test_spaxel_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.test_spaxel_pk_seq OWNER TO manga;

--
-- Name: test_spaxel_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE test_spaxel_pk_seq OWNED BY test_spaxel.pk;


--
-- Name: wavelength; Type: TABLE; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE TABLE wavelength (
    pk integer NOT NULL,
    wavelength real[],
    bintype text
);


ALTER TABLE mangadatadb.wavelength OWNER TO manga;

--
-- Name: wavelength_pk_seq; Type: SEQUENCE; Schema: mangadatadb; Owner: manga
--

CREATE SEQUENCE wavelength_pk_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE mangadatadb.wavelength_pk_seq OWNER TO manga;

--
-- Name: wavelength_pk_seq; Type: SEQUENCE OWNED BY; Schema: mangadatadb; Owner: manga
--

ALTER SEQUENCE wavelength_pk_seq OWNED BY wavelength.pk;


SET search_path = mangaauxdb, pg_catalog;

--
-- Name: pk; Type: DEFAULT; Schema: mangaauxdb; Owner: manga
--

ALTER TABLE ONLY cube_header ALTER COLUMN pk SET DEFAULT nextval('cube_header_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangaauxdb; Owner: manga
--

ALTER TABLE ONLY maskbit ALTER COLUMN pk SET DEFAULT nextval('maskbit_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangaauxdb; Owner: manga
--

ALTER TABLE ONLY maskbit_labels ALTER COLUMN pk SET DEFAULT nextval('maskbit_labels_pk_seq'::regclass);


SET search_path = mangadatadb, pg_catalog;

--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cart ALTER COLUMN pk SET DEFAULT nextval('cart_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cart_to_cube ALTER COLUMN pk SET DEFAULT nextval('cart_to_cube_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cube ALTER COLUMN pk SET DEFAULT nextval('cube_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fiber_type ALTER COLUMN pk SET DEFAULT nextval('fiber_type_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fibers ALTER COLUMN pk SET DEFAULT nextval('fibers_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fits_header_keyword ALTER COLUMN pk SET DEFAULT nextval('fits_header_keyword_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fits_header_value ALTER COLUMN pk SET DEFAULT nextval('fits_header_value_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY ifu_to_block ALTER COLUMN pk SET DEFAULT nextval('ifu_to_block_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY ifudesign ALTER COLUMN pk SET DEFAULT nextval('ifudesign_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_completion_status ALTER COLUMN pk SET DEFAULT nextval('pipeline_completion_status_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_info ALTER COLUMN pk SET DEFAULT nextval('pipeline_info_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_name ALTER COLUMN pk SET DEFAULT nextval('pipeline_name_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_stage ALTER COLUMN pk SET DEFAULT nextval('pipeline_stage_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_version ALTER COLUMN pk SET DEFAULT nextval('pipeline_version_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY rssfiber ALTER COLUMN pk SET DEFAULT nextval('rssfiber_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY sample ALTER COLUMN pk SET DEFAULT nextval('sample_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY slitblock ALTER COLUMN pk SET DEFAULT nextval('slitblock_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY spaxel ALTER COLUMN pk SET DEFAULT nextval('spaxel_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY target_type ALTER COLUMN pk SET DEFAULT nextval('target_type_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY test_rssfiber ALTER COLUMN pk SET DEFAULT nextval('test_rssfiber_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY test_spaxel ALTER COLUMN pk SET DEFAULT nextval('test_spaxel_pk_seq'::regclass);


--
-- Name: pk; Type: DEFAULT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY wavelength ALTER COLUMN pk SET DEFAULT nextval('wavelength_pk_seq'::regclass);


SET search_path = mangaauxdb, pg_catalog;

--
-- Name: cube_header_pkey; Type: CONSTRAINT; Schema: mangaauxdb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY cube_header
    ADD CONSTRAINT cube_header_pkey PRIMARY KEY (pk);


--
-- Name: maskbit_labels_pkey; Type: CONSTRAINT; Schema: mangaauxdb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY maskbit_labels
    ADD CONSTRAINT maskbit_labels_pkey PRIMARY KEY (pk);


--
-- Name: maskbit_pkey; Type: CONSTRAINT; Schema: mangaauxdb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY maskbit
    ADD CONSTRAINT maskbit_pkey PRIMARY KEY (pk);


SET search_path = mangadatadb, pg_catalog;

--
-- Name: cart_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY cart
    ADD CONSTRAINT cart_pkey PRIMARY KEY (pk);


--
-- Name: cart_to_cube_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY cart_to_cube
    ADD CONSTRAINT cart_to_cube_pkey PRIMARY KEY (pk);


--
-- Name: cube_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY cube
    ADD CONSTRAINT cube_pkey PRIMARY KEY (pk);


--
-- Name: fiber_type_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY fiber_type
    ADD CONSTRAINT fiber_type_pkey PRIMARY KEY (pk);


--
-- Name: fibers_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY fibers
    ADD CONSTRAINT fibers_pkey PRIMARY KEY (pk);


--
-- Name: fits_header_keyword_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY fits_header_keyword
    ADD CONSTRAINT fits_header_keyword_pkey PRIMARY KEY (pk);


--
-- Name: fits_header_value_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY fits_header_value
    ADD CONSTRAINT fits_header_value_pkey PRIMARY KEY (pk);


--
-- Name: ifu_to_block_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY ifu_to_block
    ADD CONSTRAINT ifu_to_block_pkey PRIMARY KEY (pk);


--
-- Name: ifudesign_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY ifudesign
    ADD CONSTRAINT ifudesign_pkey PRIMARY KEY (pk);


--
-- Name: pipeline_completion_status_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY pipeline_completion_status
    ADD CONSTRAINT pipeline_completion_status_pkey PRIMARY KEY (pk);


--
-- Name: pipeline_info_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY pipeline_info
    ADD CONSTRAINT pipeline_info_pkey PRIMARY KEY (pk);


--
-- Name: pipeline_name_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY pipeline_name
    ADD CONSTRAINT pipeline_name_pkey PRIMARY KEY (pk);


--
-- Name: pipeline_stage_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY pipeline_stage
    ADD CONSTRAINT pipeline_stage_pkey PRIMARY KEY (pk);


--
-- Name: pipeline_version_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY pipeline_version
    ADD CONSTRAINT pipeline_version_pkey PRIMARY KEY (pk);


--
-- Name: rssfiber_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY rssfiber
    ADD CONSTRAINT rssfiber_pkey PRIMARY KEY (pk);


--
-- Name: sample_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY sample
    ADD CONSTRAINT sample_pkey PRIMARY KEY (pk);


--
-- Name: slitblock_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY slitblock
    ADD CONSTRAINT slitblock_pkey PRIMARY KEY (pk);


--
-- Name: spaxel_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY spaxel
    ADD CONSTRAINT spaxel_pkey PRIMARY KEY (pk);


--
-- Name: target_type_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY target_type
    ADD CONSTRAINT target_type_pkey PRIMARY KEY (pk);


--
-- Name: test_rssfiber_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY test_rssfiber
    ADD CONSTRAINT test_rssfiber_pkey PRIMARY KEY (pk);


--
-- Name: test_spaxel_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY test_spaxel
    ADD CONSTRAINT test_spaxel_pkey PRIMARY KEY (pk);


--
-- Name: wavelength_pkey; Type: CONSTRAINT; Schema: mangadatadb; Owner: manga; Tablespace: 
--

ALTER TABLE ONLY wavelength
    ADD CONSTRAINT wavelength_pkey PRIMARY KEY (pk);


--
-- Name: rssfib_idx; Type: INDEX; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE INDEX rssfib_idx ON rssfiber USING gin (flux, ivar, mask, xpos, ypos);


--
-- Name: spaxel_idx; Type: INDEX; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE INDEX spaxel_idx ON spaxel USING gin (flux, ivar, mask);


--
-- Name: test_spaxel_idx; Type: INDEX; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE INDEX test_spaxel_idx ON test_spaxel USING gin (flux, ivar, mask);


--
-- Name: wave_idx; Type: INDEX; Schema: mangadatadb; Owner: manga; Tablespace: 
--

CREATE INDEX wave_idx ON wavelength USING gin (wavelength);


SET search_path = mangaauxdb, pg_catalog;

--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangaauxdb; Owner: manga
--

ALTER TABLE ONLY cube_header
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES mangadatadb.cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


SET search_path = mangadatadb, pg_catalog;

--
-- Name: cart_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cart_to_cube
    ADD CONSTRAINT cart_fk FOREIGN KEY (cart_pk) REFERENCES cart(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY rssfiber
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY spaxel
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fits_header_value
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY sample
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY test_rssfiber
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cart_to_cube
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: cube_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY test_spaxel
    ADD CONSTRAINT cube_fk FOREIGN KEY (cube_pk) REFERENCES cube(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: fiber_type_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fibers
    ADD CONSTRAINT fiber_type_fk FOREIGN KEY (fiber_type_pk) REFERENCES fiber_type(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: fits_header_keyword_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fits_header_value
    ADD CONSTRAINT fits_header_keyword_fk FOREIGN KEY (fits_header_keyword_pk) REFERENCES fits_header_keyword(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: ifudesign_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cube
    ADD CONSTRAINT ifudesign_fk FOREIGN KEY (ifudesign_pk) REFERENCES ifudesign(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: ifudesign_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY ifu_to_block
    ADD CONSTRAINT ifudesign_fk FOREIGN KEY (ifudesign_pk) REFERENCES ifudesign(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: ifudesign_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fibers
    ADD CONSTRAINT ifudesign_fk FOREIGN KEY (ifudesign_pk) REFERENCES ifudesign(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: pipeline_completion_status_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_info
    ADD CONSTRAINT pipeline_completion_status_fk FOREIGN KEY (pipeline_completion_status_pk) REFERENCES pipeline_completion_status(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: pipeline_info_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cube
    ADD CONSTRAINT pipeline_info_fk FOREIGN KEY (pipeline_info_pk) REFERENCES pipeline_info(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: pipeline_name_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_info
    ADD CONSTRAINT pipeline_name_fk FOREIGN KEY (pipeline_name_pk) REFERENCES pipeline_name(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: pipeline_stage_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_info
    ADD CONSTRAINT pipeline_stage_fk FOREIGN KEY (pipeline_stage_pk) REFERENCES pipeline_stage(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: pipeline_version_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY pipeline_info
    ADD CONSTRAINT pipeline_version_fk FOREIGN KEY (pipeline_version_pk) REFERENCES pipeline_version(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: slitblock_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY ifu_to_block
    ADD CONSTRAINT slitblock_fk FOREIGN KEY (slitblock_pk) REFERENCES slitblock(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: target_type_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY fibers
    ADD CONSTRAINT target_type_fk FOREIGN KEY (target_type_pk) REFERENCES target_type(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: wavelength_fk; Type: FK CONSTRAINT; Schema: mangadatadb; Owner: manga
--

ALTER TABLE ONLY cube
    ADD CONSTRAINT wavelength_fk FOREIGN KEY (wavelength_pk) REFERENCES wavelength(pk) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: functions; Type: ACL; Schema: -; Owner: manga
--

REVOKE ALL ON SCHEMA functions FROM PUBLIC;
REVOKE ALL ON SCHEMA functions FROM manga;
GRANT ALL ON SCHEMA functions TO manga;
GRANT ALL ON SCHEMA functions TO sdss;
GRANT ALL ON SCHEMA functions TO u0857802;


--
-- Name: mangaauxdb; Type: ACL; Schema: -; Owner: manga
--

REVOKE ALL ON SCHEMA mangaauxdb FROM PUBLIC;
REVOKE ALL ON SCHEMA mangaauxdb FROM manga;
GRANT ALL ON SCHEMA mangaauxdb TO manga;
GRANT ALL ON SCHEMA mangaauxdb TO sdss;
GRANT ALL ON SCHEMA mangaauxdb TO u0931042;
GRANT ALL ON SCHEMA mangaauxdb TO u0857802;
GRANT ALL ON SCHEMA mangaauxdb TO u0707758;


--
-- Name: mangadatadb; Type: ACL; Schema: -; Owner: manga
--

REVOKE ALL ON SCHEMA mangadatadb FROM PUBLIC;
REVOKE ALL ON SCHEMA mangadatadb FROM manga;
GRANT ALL ON SCHEMA mangadatadb TO manga;
GRANT ALL ON SCHEMA mangadatadb TO u0707758;
GRANT ALL ON SCHEMA mangadatadb TO u0857802;
GRANT ALL ON SCHEMA mangadatadb TO u0931042;
GRANT ALL ON SCHEMA mangadatadb TO sdss;


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


SET search_path = mangaauxdb, pg_catalog;

--
-- Name: cube_header; Type: ACL; Schema: mangaauxdb; Owner: manga
--

REVOKE ALL ON TABLE cube_header FROM PUBLIC;
REVOKE ALL ON TABLE cube_header FROM manga;
GRANT ALL ON TABLE cube_header TO manga;
GRANT ALL ON TABLE cube_header TO sdss;
GRANT ALL ON TABLE cube_header TO u0931042;
GRANT ALL ON TABLE cube_header TO u0707758;
GRANT ALL ON TABLE cube_header TO u0857802;


--
-- Name: cube_header_pk_seq; Type: ACL; Schema: mangaauxdb; Owner: manga
--

REVOKE ALL ON SEQUENCE cube_header_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE cube_header_pk_seq FROM manga;
GRANT ALL ON SEQUENCE cube_header_pk_seq TO manga;
GRANT ALL ON SEQUENCE cube_header_pk_seq TO sdss;
GRANT ALL ON SEQUENCE cube_header_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE cube_header_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE cube_header_pk_seq TO u0931042;


--
-- Name: maskbit; Type: ACL; Schema: mangaauxdb; Owner: manga
--

REVOKE ALL ON TABLE maskbit FROM PUBLIC;
REVOKE ALL ON TABLE maskbit FROM manga;
GRANT ALL ON TABLE maskbit TO manga;
GRANT ALL ON TABLE maskbit TO sdss;
GRANT ALL ON TABLE maskbit TO u0931042;
GRANT ALL ON TABLE maskbit TO u0707758;
GRANT ALL ON TABLE maskbit TO u0857802;


--
-- Name: maskbit_labels; Type: ACL; Schema: mangaauxdb; Owner: manga
--

REVOKE ALL ON TABLE maskbit_labels FROM PUBLIC;
REVOKE ALL ON TABLE maskbit_labels FROM manga;
GRANT ALL ON TABLE maskbit_labels TO manga;
GRANT ALL ON TABLE maskbit_labels TO sdss;
GRANT ALL ON TABLE maskbit_labels TO u0931042;
GRANT ALL ON TABLE maskbit_labels TO u0707758;
GRANT ALL ON TABLE maskbit_labels TO u0857802;


--
-- Name: maskbit_labels_pk_seq; Type: ACL; Schema: mangaauxdb; Owner: manga
--

REVOKE ALL ON SEQUENCE maskbit_labels_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE maskbit_labels_pk_seq FROM manga;
GRANT ALL ON SEQUENCE maskbit_labels_pk_seq TO manga;
GRANT ALL ON SEQUENCE maskbit_labels_pk_seq TO sdss;
GRANT ALL ON SEQUENCE maskbit_labels_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE maskbit_labels_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE maskbit_labels_pk_seq TO u0931042;


--
-- Name: maskbit_pk_seq; Type: ACL; Schema: mangaauxdb; Owner: manga
--

REVOKE ALL ON SEQUENCE maskbit_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE maskbit_pk_seq FROM manga;
GRANT ALL ON SEQUENCE maskbit_pk_seq TO manga;
GRANT ALL ON SEQUENCE maskbit_pk_seq TO sdss;
GRANT ALL ON SEQUENCE maskbit_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE maskbit_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE maskbit_pk_seq TO u0931042;


SET search_path = mangadatadb, pg_catalog;

--
-- Name: cart; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE cart FROM PUBLIC;
REVOKE ALL ON TABLE cart FROM manga;
GRANT ALL ON TABLE cart TO manga;
GRANT ALL ON TABLE cart TO u0707758;
GRANT ALL ON TABLE cart TO u0857802;
GRANT ALL ON TABLE cart TO u0931042;
GRANT ALL ON TABLE cart TO sdss;


--
-- Name: cart_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE cart_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE cart_pk_seq FROM manga;
GRANT ALL ON SEQUENCE cart_pk_seq TO manga;
GRANT ALL ON SEQUENCE cart_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE cart_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE cart_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE cart_pk_seq TO sdss;


--
-- Name: cart_to_cube; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE cart_to_cube FROM PUBLIC;
REVOKE ALL ON TABLE cart_to_cube FROM manga;
GRANT ALL ON TABLE cart_to_cube TO manga;
GRANT ALL ON TABLE cart_to_cube TO sdss;
GRANT ALL ON TABLE cart_to_cube TO u0857802;
GRANT ALL ON TABLE cart_to_cube TO u0707758;
GRANT ALL ON TABLE cart_to_cube TO u0931042;


--
-- Name: cart_to_cube_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE cart_to_cube_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE cart_to_cube_pk_seq FROM manga;
GRANT ALL ON SEQUENCE cart_to_cube_pk_seq TO manga;
GRANT ALL ON SEQUENCE cart_to_cube_pk_seq TO sdss;
GRANT ALL ON SEQUENCE cart_to_cube_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE cart_to_cube_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE cart_to_cube_pk_seq TO u0931042;


--
-- Name: cube; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE cube FROM PUBLIC;
REVOKE ALL ON TABLE cube FROM manga;
GRANT ALL ON TABLE cube TO manga;
GRANT ALL ON TABLE cube TO u0707758;
GRANT ALL ON TABLE cube TO u0857802;
GRANT ALL ON TABLE cube TO u0931042;
GRANT ALL ON TABLE cube TO sdss;


--
-- Name: cube_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE cube_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE cube_pk_seq FROM manga;
GRANT ALL ON SEQUENCE cube_pk_seq TO manga;
GRANT ALL ON SEQUENCE cube_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE cube_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE cube_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE cube_pk_seq TO sdss;


--
-- Name: fiber_type; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE fiber_type FROM PUBLIC;
REVOKE ALL ON TABLE fiber_type FROM manga;
GRANT ALL ON TABLE fiber_type TO manga;
GRANT ALL ON TABLE fiber_type TO u0707758;
GRANT ALL ON TABLE fiber_type TO u0857802;
GRANT ALL ON TABLE fiber_type TO u0931042;
GRANT ALL ON TABLE fiber_type TO sdss;


--
-- Name: fiber_type_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE fiber_type_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE fiber_type_pk_seq FROM manga;
GRANT ALL ON SEQUENCE fiber_type_pk_seq TO manga;
GRANT ALL ON SEQUENCE fiber_type_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE fiber_type_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE fiber_type_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE fiber_type_pk_seq TO sdss;


--
-- Name: fibers; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE fibers FROM PUBLIC;
REVOKE ALL ON TABLE fibers FROM manga;
GRANT ALL ON TABLE fibers TO manga;
GRANT ALL ON TABLE fibers TO u0707758;
GRANT ALL ON TABLE fibers TO u0857802;
GRANT ALL ON TABLE fibers TO u0931042;
GRANT ALL ON TABLE fibers TO sdss;


--
-- Name: fibers_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE fibers_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE fibers_pk_seq FROM manga;
GRANT ALL ON SEQUENCE fibers_pk_seq TO manga;
GRANT ALL ON SEQUENCE fibers_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE fibers_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE fibers_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE fibers_pk_seq TO sdss;


--
-- Name: fits_header_keyword; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE fits_header_keyword FROM PUBLIC;
REVOKE ALL ON TABLE fits_header_keyword FROM manga;
GRANT ALL ON TABLE fits_header_keyword TO manga;
GRANT ALL ON TABLE fits_header_keyword TO u0707758;
GRANT ALL ON TABLE fits_header_keyword TO u0857802;
GRANT ALL ON TABLE fits_header_keyword TO u0931042;
GRANT ALL ON TABLE fits_header_keyword TO sdss;


--
-- Name: fits_header_keyword_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE fits_header_keyword_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE fits_header_keyword_pk_seq FROM manga;
GRANT ALL ON SEQUENCE fits_header_keyword_pk_seq TO manga;
GRANT ALL ON SEQUENCE fits_header_keyword_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE fits_header_keyword_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE fits_header_keyword_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE fits_header_keyword_pk_seq TO sdss;


--
-- Name: fits_header_value; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE fits_header_value FROM PUBLIC;
REVOKE ALL ON TABLE fits_header_value FROM manga;
GRANT ALL ON TABLE fits_header_value TO manga;
GRANT ALL ON TABLE fits_header_value TO u0707758;
GRANT ALL ON TABLE fits_header_value TO u0857802;
GRANT ALL ON TABLE fits_header_value TO u0931042;
GRANT ALL ON TABLE fits_header_value TO sdss;


--
-- Name: fits_header_value_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE fits_header_value_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE fits_header_value_pk_seq FROM manga;
GRANT ALL ON SEQUENCE fits_header_value_pk_seq TO manga;
GRANT ALL ON SEQUENCE fits_header_value_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE fits_header_value_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE fits_header_value_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE fits_header_value_pk_seq TO sdss;


--
-- Name: ifu_to_block; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE ifu_to_block FROM PUBLIC;
REVOKE ALL ON TABLE ifu_to_block FROM manga;
GRANT ALL ON TABLE ifu_to_block TO manga;
GRANT ALL ON TABLE ifu_to_block TO u0707758;
GRANT ALL ON TABLE ifu_to_block TO u0857802;
GRANT ALL ON TABLE ifu_to_block TO u0931042;
GRANT ALL ON TABLE ifu_to_block TO sdss;


--
-- Name: ifu_to_block_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE ifu_to_block_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE ifu_to_block_pk_seq FROM manga;
GRANT ALL ON SEQUENCE ifu_to_block_pk_seq TO manga;
GRANT ALL ON SEQUENCE ifu_to_block_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE ifu_to_block_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE ifu_to_block_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE ifu_to_block_pk_seq TO sdss;


--
-- Name: ifudesign; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE ifudesign FROM PUBLIC;
REVOKE ALL ON TABLE ifudesign FROM manga;
GRANT ALL ON TABLE ifudesign TO manga;
GRANT ALL ON TABLE ifudesign TO u0707758;
GRANT ALL ON TABLE ifudesign TO u0857802;
GRANT ALL ON TABLE ifudesign TO u0931042;
GRANT ALL ON TABLE ifudesign TO sdss;


--
-- Name: ifudesign_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE ifudesign_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE ifudesign_pk_seq FROM manga;
GRANT ALL ON SEQUENCE ifudesign_pk_seq TO manga;
GRANT ALL ON SEQUENCE ifudesign_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE ifudesign_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE ifudesign_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE ifudesign_pk_seq TO sdss;


--
-- Name: pipeline_completion_status; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE pipeline_completion_status FROM PUBLIC;
REVOKE ALL ON TABLE pipeline_completion_status FROM manga;
GRANT ALL ON TABLE pipeline_completion_status TO manga;
GRANT ALL ON TABLE pipeline_completion_status TO u0707758;
GRANT ALL ON TABLE pipeline_completion_status TO u0857802;
GRANT ALL ON TABLE pipeline_completion_status TO u0931042;
GRANT ALL ON TABLE pipeline_completion_status TO sdss;


--
-- Name: pipeline_completion_status_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE pipeline_completion_status_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE pipeline_completion_status_pk_seq FROM manga;
GRANT ALL ON SEQUENCE pipeline_completion_status_pk_seq TO manga;
GRANT ALL ON SEQUENCE pipeline_completion_status_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE pipeline_completion_status_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE pipeline_completion_status_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE pipeline_completion_status_pk_seq TO sdss;


--
-- Name: pipeline_info; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE pipeline_info FROM PUBLIC;
REVOKE ALL ON TABLE pipeline_info FROM manga;
GRANT ALL ON TABLE pipeline_info TO manga;
GRANT ALL ON TABLE pipeline_info TO u0707758;
GRANT ALL ON TABLE pipeline_info TO u0857802;
GRANT ALL ON TABLE pipeline_info TO u0931042;
GRANT ALL ON TABLE pipeline_info TO sdss;


--
-- Name: pipeline_info_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE pipeline_info_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE pipeline_info_pk_seq FROM manga;
GRANT ALL ON SEQUENCE pipeline_info_pk_seq TO manga;
GRANT ALL ON SEQUENCE pipeline_info_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE pipeline_info_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE pipeline_info_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE pipeline_info_pk_seq TO sdss;


--
-- Name: pipeline_name; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE pipeline_name FROM PUBLIC;
REVOKE ALL ON TABLE pipeline_name FROM manga;
GRANT ALL ON TABLE pipeline_name TO manga;
GRANT ALL ON TABLE pipeline_name TO u0707758;
GRANT ALL ON TABLE pipeline_name TO u0857802;
GRANT ALL ON TABLE pipeline_name TO u0931042;
GRANT ALL ON TABLE pipeline_name TO sdss;


--
-- Name: pipeline_name_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE pipeline_name_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE pipeline_name_pk_seq FROM manga;
GRANT ALL ON SEQUENCE pipeline_name_pk_seq TO manga;
GRANT ALL ON SEQUENCE pipeline_name_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE pipeline_name_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE pipeline_name_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE pipeline_name_pk_seq TO sdss;


--
-- Name: pipeline_stage; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE pipeline_stage FROM PUBLIC;
REVOKE ALL ON TABLE pipeline_stage FROM manga;
GRANT ALL ON TABLE pipeline_stage TO manga;
GRANT ALL ON TABLE pipeline_stage TO u0707758;
GRANT ALL ON TABLE pipeline_stage TO u0857802;
GRANT ALL ON TABLE pipeline_stage TO u0931042;
GRANT ALL ON TABLE pipeline_stage TO sdss;


--
-- Name: pipeline_stage_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE pipeline_stage_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE pipeline_stage_pk_seq FROM manga;
GRANT ALL ON SEQUENCE pipeline_stage_pk_seq TO manga;
GRANT ALL ON SEQUENCE pipeline_stage_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE pipeline_stage_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE pipeline_stage_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE pipeline_stage_pk_seq TO sdss;


--
-- Name: pipeline_version; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE pipeline_version FROM PUBLIC;
REVOKE ALL ON TABLE pipeline_version FROM manga;
GRANT ALL ON TABLE pipeline_version TO manga;
GRANT ALL ON TABLE pipeline_version TO u0707758;
GRANT ALL ON TABLE pipeline_version TO u0857802;
GRANT ALL ON TABLE pipeline_version TO u0931042;
GRANT ALL ON TABLE pipeline_version TO sdss;


--
-- Name: pipeline_version_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE pipeline_version_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE pipeline_version_pk_seq FROM manga;
GRANT ALL ON SEQUENCE pipeline_version_pk_seq TO manga;
GRANT ALL ON SEQUENCE pipeline_version_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE pipeline_version_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE pipeline_version_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE pipeline_version_pk_seq TO sdss;


--
-- Name: rssfiber; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE rssfiber FROM PUBLIC;
REVOKE ALL ON TABLE rssfiber FROM manga;
GRANT ALL ON TABLE rssfiber TO manga;
GRANT ALL ON TABLE rssfiber TO u0707758;
GRANT ALL ON TABLE rssfiber TO u0857802;
GRANT ALL ON TABLE rssfiber TO u0931042;
GRANT ALL ON TABLE rssfiber TO sdss;


--
-- Name: rssfiber_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE rssfiber_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE rssfiber_pk_seq FROM manga;
GRANT ALL ON SEQUENCE rssfiber_pk_seq TO manga;
GRANT ALL ON SEQUENCE rssfiber_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE rssfiber_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE rssfiber_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE rssfiber_pk_seq TO sdss;


--
-- Name: sample; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE sample FROM PUBLIC;
REVOKE ALL ON TABLE sample FROM manga;
GRANT ALL ON TABLE sample TO manga;
GRANT ALL ON TABLE sample TO u0707758;
GRANT ALL ON TABLE sample TO u0857802;
GRANT ALL ON TABLE sample TO u0931042;
GRANT ALL ON TABLE sample TO sdss;


--
-- Name: sample_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE sample_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE sample_pk_seq FROM manga;
GRANT ALL ON SEQUENCE sample_pk_seq TO manga;
GRANT ALL ON SEQUENCE sample_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE sample_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE sample_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE sample_pk_seq TO sdss;


--
-- Name: slitblock; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE slitblock FROM PUBLIC;
REVOKE ALL ON TABLE slitblock FROM manga;
GRANT ALL ON TABLE slitblock TO manga;
GRANT ALL ON TABLE slitblock TO u0707758;
GRANT ALL ON TABLE slitblock TO u0857802;
GRANT ALL ON TABLE slitblock TO u0931042;
GRANT ALL ON TABLE slitblock TO sdss;


--
-- Name: slitblock_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE slitblock_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE slitblock_pk_seq FROM manga;
GRANT ALL ON SEQUENCE slitblock_pk_seq TO manga;
GRANT ALL ON SEQUENCE slitblock_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE slitblock_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE slitblock_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE slitblock_pk_seq TO sdss;


--
-- Name: spaxel; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE spaxel FROM PUBLIC;
REVOKE ALL ON TABLE spaxel FROM manga;
GRANT ALL ON TABLE spaxel TO manga;
GRANT ALL ON TABLE spaxel TO u0707758;
GRANT ALL ON TABLE spaxel TO u0857802;
GRANT ALL ON TABLE spaxel TO u0931042;
GRANT ALL ON TABLE spaxel TO sdss;


--
-- Name: spaxel_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE spaxel_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE spaxel_pk_seq FROM manga;
GRANT ALL ON SEQUENCE spaxel_pk_seq TO manga;
GRANT ALL ON SEQUENCE spaxel_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE spaxel_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE spaxel_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE spaxel_pk_seq TO sdss;


--
-- Name: target_type; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE target_type FROM PUBLIC;
REVOKE ALL ON TABLE target_type FROM manga;
GRANT ALL ON TABLE target_type TO manga;
GRANT ALL ON TABLE target_type TO u0707758;
GRANT ALL ON TABLE target_type TO u0857802;
GRANT ALL ON TABLE target_type TO u0931042;
GRANT ALL ON TABLE target_type TO sdss;


--
-- Name: target_type_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE target_type_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE target_type_pk_seq FROM manga;
GRANT ALL ON SEQUENCE target_type_pk_seq TO manga;
GRANT ALL ON SEQUENCE target_type_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE target_type_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE target_type_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE target_type_pk_seq TO sdss;


--
-- Name: test_rssfiber; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE test_rssfiber FROM PUBLIC;
REVOKE ALL ON TABLE test_rssfiber FROM manga;
GRANT ALL ON TABLE test_rssfiber TO manga;
GRANT ALL ON TABLE test_rssfiber TO u0707758;
GRANT ALL ON TABLE test_rssfiber TO u0857802;
GRANT ALL ON TABLE test_rssfiber TO u0931042;
GRANT ALL ON TABLE test_rssfiber TO sdss;


--
-- Name: test_rssfiber_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE test_rssfiber_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE test_rssfiber_pk_seq FROM manga;
GRANT ALL ON SEQUENCE test_rssfiber_pk_seq TO manga;
GRANT ALL ON SEQUENCE test_rssfiber_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE test_rssfiber_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE test_rssfiber_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE test_rssfiber_pk_seq TO sdss;


--
-- Name: test_spaxel; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE test_spaxel FROM PUBLIC;
REVOKE ALL ON TABLE test_spaxel FROM manga;
GRANT ALL ON TABLE test_spaxel TO manga;
GRANT ALL ON TABLE test_spaxel TO sdss;
GRANT ALL ON TABLE test_spaxel TO u0857802;
GRANT ALL ON TABLE test_spaxel TO u0707758;
GRANT ALL ON TABLE test_spaxel TO u0931042;


--
-- Name: test_spaxel_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE test_spaxel_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE test_spaxel_pk_seq FROM manga;
GRANT ALL ON SEQUENCE test_spaxel_pk_seq TO manga;
GRANT ALL ON SEQUENCE test_spaxel_pk_seq TO sdss;
GRANT ALL ON SEQUENCE test_spaxel_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE test_spaxel_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE test_spaxel_pk_seq TO u0931042;


--
-- Name: wavelength; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON TABLE wavelength FROM PUBLIC;
REVOKE ALL ON TABLE wavelength FROM manga;
GRANT ALL ON TABLE wavelength TO manga;
GRANT ALL ON TABLE wavelength TO u0707758;
GRANT ALL ON TABLE wavelength TO u0857802;
GRANT ALL ON TABLE wavelength TO u0931042;
GRANT ALL ON TABLE wavelength TO sdss;


--
-- Name: wavelength_pk_seq; Type: ACL; Schema: mangadatadb; Owner: manga
--

REVOKE ALL ON SEQUENCE wavelength_pk_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE wavelength_pk_seq FROM manga;
GRANT ALL ON SEQUENCE wavelength_pk_seq TO manga;
GRANT ALL ON SEQUENCE wavelength_pk_seq TO u0707758;
GRANT ALL ON SEQUENCE wavelength_pk_seq TO u0857802;
GRANT ALL ON SEQUENCE wavelength_pk_seq TO u0931042;
GRANT ALL ON SEQUENCE wavelength_pk_seq TO sdss;


--
-- PostgreSQL database dump complete
--

