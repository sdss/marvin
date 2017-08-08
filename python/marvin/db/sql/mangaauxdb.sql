/*

mangaAuxDB schema version

Stores auxiliary information for mangaDataDB.

Create March,2015 - B. Cherinka

*/

CREATE SCHEMA mangaauxdb;

SET search_path TO mangaauxdb;

CREATE TABLE mangaauxdb.cube_header (pk serial PRIMARY KEY NOT NULL, header JSON, cube_pk INTEGER);

CREATE TABLE mangaauxdb.maskbit_labels (pk serial PRIMARY KEY NOT NULL, flag TEXT, maskbit INTEGER, labels JSON);

CREATE TABLE mangaauxdb.maskbit (pk serial PRIMARY KEY NOT NULL, flag TEXT, bit INTEGER, label TEXT, description TEXT);

ALTER TABLE ONLY mangaauxdb.cube_header
    ADD CONSTRAINT cube_fk
    FOREIGN KEY (cube_pk) REFERENCES mangadatadb.cube(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;


/* Functions related to MaNGA Aux DB */
SET search_path TO functions;

/* return a label for a given mask bit */
CREATE OR REPLACE FUNCTION getmasklabel(maskbit integer, flagname text) RETURNS text[]
    LANGUAGE plpgsql IMMUTABLE
    AS $$

DECLARE labels text[];
BEGIN
	with x as (select string_to_array(reverse(maskbit::bit(10)::text)::varbit::text,null) as b)
	select array_agg(m.label) into labels
	from (select b, generate_subscripts(b,1) as i from x) as y, mangaauxdb.maskbit as m
	where b[i]='1' and m.bit=(i-1) and m.flag=flagname;
	return labels;
END; $$;

/* return array of labels for a set of mask bits */
CREATE OR REPLACE FUNCTION getmasklabels(maskbit integer[], flagname text) RETURNS SETOF text[]
    LANGUAGE plpgsql IMMUTABLE
    AS $$

DECLARE mbit integer;
BEGIN
	FOREACH mbit IN ARRAY maskbit
	LOOP
		return query (select getmasklabel(mbit, flagname));
	END LOOP;
END; $$;


