
/*

New MaNGA DAP DB schema - Sept 7th 2016

This is an alternate schema where all the value_types are actually new
columns in the spaxelprop table

procedure to load the dapdb
- no indexes, foreign keys
- load Metadata
- load data
- add foreign keys
- add indexes
 */


create schema mangadapdb;

set search_path to mangadapdb;

create table mangadapdb.file (pk serial primary key not null, filename text, filepath text, num_ext integer, filetype_pk integer, structure_pk integer, cube_pk integer, pipeline_info_pk integer);

create table mangadapdb.filetype (pk serial primary key not null, value text);

create table mangadapdb.current_default (pk serial primary key not null, filename text, filepath text, file_pk integer);

create table mangadapdb.hdu (pk serial primary key not null, extname_pk integer, exttype_pk integer, extno integer, file_pk integer);

create table mangadapdb.hdu_to_header_value (pk serial primary key not null, hdu_pk integer, header_value_pk integer);

create table mangadapdb.header_value (pk serial primary key not null, value text, index integer, comment text, header_keyword_pk integer);

create table mangadapdb.header_keyword (pk serial primary key not null, name text);

create table mangadapdb.exttype (pk serial primary key not null, name text);

create table mangadapdb.extname (pk serial primary key not null, name text);

create table mangadapdb.hdu_to_extcol (pk serial primary key not null, hdu_pk integer, extcol_pk integer);

create table mangadapdb.extcol (pk serial primary key not null, name text);

create table mangadapdb.structure (pk serial primary key not null, binmode_pk integer, bintype_pk integer, template_kin_pk integer, template_pop_pk integer, executionplan_pk integer);

create table mangadapdb.binid (pk integer primary key not null, id integer);

create table mangadapdb.executionplan (pk serial primary key not null, id integer, comments text);

create table mangadapdb.template (pk serial primary key not null, name text, id integer);

create table mangadapdb.binmode (pk serial primary key not null, name text);

create table mangadapdb.bintype (pk serial primary key not null, name text);

create table mangadapdb.spaxelprop (pk bigserial primary key not null, file_pk integer, spaxel_index integer, binid_pk integer, x integer, y integer);

create table mangadapdb.spaxelprop5 (pk bigserial primary key not null, file_pk integer, spaxel_index integer, binid_pk integer, x integer, y integer);

create table mangadapdb.spaxelprop6 (pk bigserial primary key not null, file_pk integer, spaxel_index integer, binid_pk integer, x integer, y integer);

create table mangadapdb.spaxelprop7 (pk bigserial primary key not null, file_pk integer, spaxel_index integer, binid_pk integer, x integer, y integer);

create table mangadapdb.spaxelprop8 (pk bigserial primary key not null, file_pk integer, spaxel_index integer, binid_pk integer, x integer, y integer);

create table mangadapdb.modelcube (pk serial primary key not null, file_pk integer);

create table mangadapdb.modelspaxel (pk serial primary key not null, flux real[], ivar real[], mask integer[], model real[],
    emline double precision[], emline_base real[], emline_mask integer[], x integer, y integer, modelcube_pk integer);

create table mangadapdb.redcorr (pk serial primary key not null, value double precision[], modelcube_pk integer);

create table mangadapdb.dapall (pk serial primary key not null, file_pk integer);

ALTER TABLE ONLY mangadapdb.file
    ADD CONSTRAINT cube_fk
    FOREIGN KEY (cube_pk) REFERENCES mangadatadb.cube(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.file
    ADD CONSTRAINT pipeline_info_fk
    FOREIGN KEY (pipeline_info_pk) REFERENCES mangadatadb.pipeline_info(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.file
    ADD CONSTRAINT filetype_fk
    FOREIGN KEY (filetype_pk) REFERENCES mangadapdb.filetype(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.current_default
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.hdu
    ADD CONSTRAINT extname_fk
    FOREIGN KEY (extname_pk) REFERENCES mangadapdb.extname(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.hdu
    ADD CONSTRAINT exttype_fk
    FOREIGN KEY (exttype_pk) REFERENCES mangadapdb.exttype(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.hdu
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.hdu_to_extcol
    ADD CONSTRAINT hdu_fk
    FOREIGN KEY (hdu_pk) REFERENCES mangadapdb.hdu(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.hdu_to_extcol
    ADD CONSTRAINT extcol_fk
    FOREIGN KEY (extcol_pk) REFERENCES mangadapdb.extcol(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.hdu_to_header_value
    ADD CONSTRAINT hdu_fk
    FOREIGN KEY (hdu_pk) REFERENCES mangadapdb.hdu(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.hdu_to_header_value
    ADD CONSTRAINT header_value_fk
    FOREIGN KEY (header_value_pk) REFERENCES mangadapdb.header_value(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.header_value
    ADD CONSTRAINT header_keyword_fk
    FOREIGN KEY (header_keyword_pk) REFERENCES mangadapdb.header_keyword(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.structure
    ADD CONSTRAINT binmode_fk
    FOREIGN KEY (binmode_pk) REFERENCES mangadapdb.binmode(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.structure
    ADD CONSTRAINT bintype_fk
    FOREIGN KEY (bintype_pk) REFERENCES mangadapdb.bintype(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.structure
    ADD CONSTRAINT template_kin_fk
    FOREIGN KEY (template_kin_pk) REFERENCES mangadapdb.template(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.structure
    ADD CONSTRAINT template_pop_fk
    FOREIGN KEY (template_pop_pk) REFERENCES mangadapdb.template(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.structure
    ADD CONSTRAINT executionplan_fk
    FOREIGN KEY (executionplan_pk) REFERENCES mangadapdb.executionplan(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.file
    ADD CONSTRAINT structure_fk
    FOREIGN KEY (structure_pk) REFERENCES mangadapdb.structure(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.modelcube
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.redcorr
    ADD CONSTRAINT modelcube_fk
    FOREIGN KEY (modelcube_pk) REFERENCES mangadapdb.modelcube(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.modelspaxel
    ADD CONSTRAINT modelcube_fk
    FOREIGN KEY (modelcube_pk) REFERENCES mangadapdb.modelcube(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.dapall
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop
    ADD CONSTRAINT binid_fk
    FOREIGN KEY (binid_pk) REFERENCES mangadapdb.binid(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop5
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop5
    ADD CONSTRAINT binid_fk
    FOREIGN KEY (binid_pk) REFERENCES mangadapdb.binid(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop6
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop6
    ADD CONSTRAINT binid_fk
    FOREIGN KEY (binid_pk) REFERENCES mangadapdb.binid(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop7
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.spaxelprop8
    ADD CONSTRAINT file_fk
    FOREIGN KEY (file_pk) REFERENCES mangadapdb.file(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

CREATE INDEX CONCURRENTLY cube_pk_idx ON mangadapdb.file using BTREE(cube_pk);
CREATE INDEX CONCURRENTLY pipeline_info_pk_idx ON mangadapdb.file using BTREE(pipeline_info_pk);
CREATE INDEX CONCURRENTLY extname_pk_idx ON mangadapdb.hdu using BTREE(extname_pk);
CREATE INDEX CONCURRENTLY exttype_pk_idx ON mangadapdb.hdu using BTREE(exttype_pk);
CREATE INDEX CONCURRENTLY file_pk_idx ON mangadapdb.hdu using BTREE(file_pk);
CREATE INDEX CONCURRENTLY hdu_pk_idx ON mangadapdb.hdu_to_header_value using BTREE(hdu_pk);
CREATE INDEX CONCURRENTLY header_value_pk_idx ON mangadapdb.hdu_to_header_value using BTREE(header_value_pk);
CREATE INDEX CONCURRENTLY header_keyword_pk_idx ON mangadapdb.header_value using BTREE(header_keyword_pk);
CREATE INDEX CONCURRENTLY id_idx ON mangadapdb.binid using BTREE(id);

CREATE INDEX CONCURRENTLY binid_idx ON mangadapdb.spaxelprop using BTREE(binid);
CREATE INDEX CONCURRENTLY file_pk_idx ON mangadapdb.spaxelprop using BTREE(file_pk);
CREATE INDEX CONCURRENTLY spaxel_index_idx ON mangadapdb.spaxelprop using BTREE(spaxel_index);
CREATE INDEX CONCURRENTLY emline_gflux_ha_idx ON mangadapdb.spaxelprop using BTREE(emline_gflux_ha_6564);
CREATE INDEX CONCURRENTLY emline_gflux_hb_idx ON mangadapdb.spaxelprop using BTREE(emline_gflux_hb_4862);
CREATE INDEX CONCURRENTLY emline_gflux_oiii_idx ON mangadapdb.spaxelprop using BTREE(emline_gflux_oiii_5008);
CREATE INDEX CONCURRENTLY emline_gflux_sii_idx ON mangadapdb.spaxelprop using BTREE(emline_gflux_sii_6718);
CREATE INDEX CONCURRENTLY emline_gflux_oii_idx ON mangadapdb.spaxelprop using BTREE(emline_gflux_oiid_3728);
CREATE INDEX CONCURRENTLY emline_gflux_nii_idx ON mangadapdb.spaxelprop using BTREE(emline_gflux_nii_6585);

CREATE INDEX CONCURRENTLY binid5_idx ON mangadapdb.spaxelprop5 using BTREE(binid);
CREATE INDEX CONCURRENTLY file5_pk_idx ON mangadapdb.spaxelprop5 using BTREE(file_pk);
CREATE INDEX CONCURRENTLY spaxel5_index_idx ON mangadapdb.spaxelprop5 using BTREE(spaxel_index);
CREATE INDEX CONCURRENTLY emline5_gflux_ha_idx ON mangadapdb.spaxelprop5 using BTREE(emline_gflux_ha_6564);
CREATE INDEX CONCURRENTLY emline5_gflux_hb_idx ON mangadapdb.spaxelprop5 using BTREE(emline_gflux_hb_4862);
CREATE INDEX CONCURRENTLY emline5_gflux_oiii_idx ON mangadapdb.spaxelprop5 using BTREE(emline_gflux_oiii_5008);
CREATE INDEX CONCURRENTLY emline5_gflux_sii_idx ON mangadapdb.spaxelprop5 using BTREE(emline_gflux_sii_6718);
CREATE INDEX CONCURRENTLY emline5_gflux_oii_idx ON mangadapdb.spaxelprop5 using BTREE(emline_gflux_oiid_3728);
CREATE INDEX CONCURRENTLY emline5_gflux_nii_idx ON mangadapdb.spaxelprop5 using BTREE(emline_gflux_nii_6585);

CREATE INDEX CONCURRENTLY binid6_idx ON mangadapdb.spaxelprop6 using BTREE(binid);
CREATE INDEX CONCURRENTLY file6_pk_idx ON mangadapdb.spaxelprop6 using BTREE(file_pk);
CREATE INDEX CONCURRENTLY spaxel6_index_idx ON mangadapdb.spaxelprop6 using BTREE(spaxel_index);
CREATE INDEX CONCURRENTLY emline6_gflux_ha_idx ON mangadapdb.spaxelprop6 using BTREE(emline_gflux_ha_6564);
CREATE INDEX CONCURRENTLY emline6_gflux_hb_idx ON mangadapdb.spaxelprop6 using BTREE(emline_gflux_hb_4862);
CREATE INDEX CONCURRENTLY emline6_gflux_oiii_idx ON mangadapdb.spaxelprop6 using BTREE(emline_gflux_oiii_5008);
CREATE INDEX CONCURRENTLY emline6_gflux_sii_idx ON mangadapdb.spaxelprop6 using BTREE(emline_gflux_sii_6718);
CREATE INDEX CONCURRENTLY emline6_gflux_oii_idx ON mangadapdb.spaxelprop6 using BTREE(emline_gflux_oiid_3728);
CREATE INDEX CONCURRENTLY emline6_gflux_nii_idx ON mangadapdb.spaxelprop6 using BTREE(emline_gflux_nii_6585);

CREATE INDEX CONCURRENTLY binid7_idx ON mangadapdb.spaxelprop7 using BTREE(binid);
CREATE INDEX CONCURRENTLY file7_pk_idx ON mangadapdb.spaxelprop7 using BTREE(file_pk);
CREATE INDEX CONCURRENTLY spaxel7_index_idx ON mangadapdb.spaxelprop7 using BTREE(spaxel_index);
create index concurrently spx7_x_idx on mangadapdb.spaxelprop7 using btree(x);
create index concurrently spx7_y_idx on mangadapdb.spaxelprop7 using btree(y);
CREATE INDEX CONCURRENTLY emline7_gflux_ha_idx ON mangadapdb.spaxelprop7 using BTREE(emline_gflux_ha_6564);
CREATE INDEX CONCURRENTLY emline7_gflux_hb_idx ON mangadapdb.spaxelprop7 using BTREE(emline_gflux_hb_4862);
CREATE INDEX CONCURRENTLY emline7_gflux_oiii_idx ON mangadapdb.spaxelprop7 using BTREE(emline_gflux_oiii_5008);
CREATE INDEX CONCURRENTLY emline7_gflux_sii_idx ON mangadapdb.spaxelprop7 using BTREE(emline_gflux_sii_6718);
CREATE INDEX CONCURRENTLY emline7_gflux_oii_idx ON mangadapdb.spaxelprop7 using BTREE(emline_gflux_oiid_3728);
CREATE INDEX CONCURRENTLY emline7_gflux_nii_idx ON mangadapdb.spaxelprop7 using BTREE(emline_gflux_nii_6585);

CREATE INDEX CONCURRENTLY binid8_idx ON mangadapdb.spaxelprop8 using BTREE(binid);
CREATE INDEX CONCURRENTLY file8_pk_idx ON mangadapdb.spaxelprop8 using BTREE(file_pk);
CREATE INDEX CONCURRENTLY spaxel8_index_idx ON mangadapdb.spaxelprop8 using BTREE(spaxel_index);
create index concurrently spx8_x_idx on mangadapdb.spaxelprop8 using btree(x);
create index concurrently spx8_y_idx on mangadapdb.spaxelprop8 using btree(y);
CREATE INDEX CONCURRENTLY emline8_gflux_ha_idx ON mangadapdb.spaxelprop8 using BTREE(emline_gflux_ha_6564);
CREATE INDEX CONCURRENTLY emline8_gflux_hb_idx ON mangadapdb.spaxelprop8 using BTREE(emline_gflux_hb_4862);
CREATE INDEX CONCURRENTLY emline8_gflux_oiii_idx ON mangadapdb.spaxelprop8 using BTREE(emline_gflux_oiii_5008);
CREATE INDEX CONCURRENTLY emline8_gflux_sii_idx ON mangadapdb.spaxelprop8 using BTREE(emline_gflux_sii_6718);
CREATE INDEX CONCURRENTLY emline8_gflux_oii_idx ON mangadapdb.spaxelprop8 using BTREE(emline_gflux_oiid_3728);
CREATE INDEX CONCURRENTLY emline8_gflux_nii_idx ON mangadapdb.spaxelprop8 using BTREE(emline_gflux_nii_6585);

CREATE INDEX CONCURRENTLY mc_file_pk_idx ON mangadapdb.modelcube using BTREE(file_pk);
CREATE INDEX CONCURRENTLY rc_mc_pk_idx ON mangadapdb.redcorr using BTREE(modelcube_pk);
CREATE INDEX CONCURRENTLY mc_pk_idx ON mangadapdb.modelspaxel using BTREE(modelcube_pk);
CREATE INDEX CONCURRENTLY ms_x_idx ON mangadapdb.modelspaxel using BTREE(x);
CREATE INDEX CONCURRENTLY ms_y_idx ON mangadapdb.modelspaxel using BTREE(y);

CREATE INDEX CONCURRENTLY dapall_file_pk_idx ON mangadapdb.dapall using BTREE(file_pk);

# CleanSpaxelProp after the initial load of SpaxelProp.  Run these only after populating the spaxeprop tables with
# the new columns and data for each MPL.
#
-- # MPL-4
-- create table mangadapdb.cleanspaxelprop as select s.* from mangadapdb.spaxelprop as s where s.binid != -1;
-- alter table mangadapdb.cleanspaxelprop add constraint file_fk foreign key (file_pk) references mangadapdb.file(pk);

-- CREATE INDEX CONCURRENTLY clean_binid_idx ON mangadapdb.cleanspaxelprop using BTREE(binid);
-- CREATE INDEX CONCURRENTLY clean_file_pk_idx ON mangadapdb.cleanspaxelprop using BTREE(file_pk);
-- CREATE INDEX CONCURRENTLY clean_spaxel_index_idx ON mangadapdb.cleanspaxelprop using BTREE(spaxel_index);
-- CREATE INDEX CONCURRENTLY clean_emline_gflux_ha_idx ON mangadapdb.cleanspaxelprop using BTREE(emline_gflux_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline_gflux_hb_idx ON mangadapdb.cleanspaxelprop using BTREE(emline_gflux_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline_gflux_oiii_idx ON mangadapdb.cleanspaxelprop using BTREE(emline_gflux_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline_gflux_sii_idx ON mangadapdb.cleanspaxelprop using BTREE(emline_gflux_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline_gflux_oii_idx ON mangadapdb.cleanspaxelprop using BTREE(emline_gflux_oiid_3728);
-- CREATE INDEX CONCURRENTLY clean_emline_gflux_nii_idx ON mangadapdb.cleanspaxelprop using BTREE(emline_gflux_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_emline_ew_ha_idx ON mangadapdb.cleanspaxelprop using BTREE(emline_ew_ha_6564);

-- # MPL-5
-- create table mangadapdb.cleanspaxelprop5 as select s.* from mangadapdb.spaxelprop5 as s where s.binid != -1;
-- alter table mangadapdb.cleanspaxelprop5 add constraint file_fk foreign key (file_pk) references mangadapdb.file(pk);

-- CREATE INDEX CONCURRENTLY clean_binid5_pk_idx ON mangadapdb.cleanspaxelprop5 using BTREE(binid);
-- CREATE INDEX CONCURRENTLY clean_file5_pk_idx ON mangadapdb.cleanspaxelprop5 using BTREE(file_pk);
-- CREATE INDEX CONCURRENTLY clean_spaxel5_index_idx ON mangadapdb.cleanspaxelprop5 using BTREE(spaxel_index);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ha_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_hb_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_oiii_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_sii_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_oii_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_oiid_3728);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_nii_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_stvel5_idx ON mangadapdb.cleanspaxelprop5 using BTREE(stellar_vel);
-- CREATE INDEX CONCURRENTLY clean_d40005_idx ON mangadapdb.cleanspaxelprop5 using BTREE(specindex_d4000);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_oi_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_siia_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_sii_6732);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_ha_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_hb_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_oiii_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_sii_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_oiid_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_oiid_3728);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_nii_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_oi_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline5_gflux_ivar_siia_idx ON mangadapdb.cleanspaxelprop5 using BTREE(emline_gflux_ivar_sii_6732);


-- # MPL-6

-- create table mangadapdb.cleanspaxelprop6 as select s.* from mangadapdb.spaxelprop6 as s
--     where (s.binid_binned_spectra != -1 and s.binid_stellar_continua != -1 and s.binid_spectral_indices != -1
--         and s.binid_em_line_moments != -1 and s.binid_em_line_models != -1);
-- alter table mangadapdb.cleanspaxelprop6 add constraint file_fk foreign key (file_pk) references mangadapdb.file(pk);

-- CREATE INDEX CONCURRENTLY clean_binid6_pk_idx ON mangadapdb.cleanspaxelprop6 using BTREE(binid);
-- CREATE INDEX CONCURRENTLY clean_file6_pk_idx ON mangadapdb.cleanspaxelprop6 using BTREE(file_pk);
-- CREATE INDEX CONCURRENTLY clean_spaxel6_index_idx ON mangadapdb.cleanspaxelprop6 using BTREE(spaxel_index);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ha_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_hb_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_oiii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_sii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_oii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_oiia_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_nii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_stvel6_idx ON mangadapdb.cleanspaxelprop6 using BTREE(stellar_vel);
-- CREATE INDEX CONCURRENTLY clean_d40006_idx ON mangadapdb.cleanspaxelprop6 using BTREE(specindex_d4000);

-- CREATE INDEX CONCURRENTLY clean_stsig6_idx ON mangadapdb.cleanspaxelprop6 using BTREE(stellar_sigma);
-- CREATE INDEX CONCURRENTLY clean_emline6_gew_ha_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gew_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline6_gew_hb_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gew_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline6_gew_oiii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gew_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline6_gew_sii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gew_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline6_gew_oii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gew_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline6_gew_oiia_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gew_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline6_gew_nii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gew_nii_6585);

-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_oi_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_siia_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_sii_6732);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_ha_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_hb_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_oiii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_sii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_oii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_oiia_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_nii_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_oi_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline6_gflux_ivar_siia_idx ON mangadapdb.cleanspaxelprop6 using BTREE(emline_gflux_ivar_sii_6732);

-- # MPL-7

-- create table mangadapdb.cleanspaxelprop7 as select s.* from mangadapdb.spaxelprop7 as s
--     where (s.binid_binned_spectra != -1 and s.binid_stellar_continua != -1 and s.binid_spectral_indices != -1
--         and s.binid_em_line_moments != -1 and s.binid_em_line_models != -1);
-- alter table mangadapdb.cleanspaxelprop7 add constraint file_fk foreign key (file_pk) references mangadapdb.file(pk);

-- CREATE INDEX CONCURRENTLY clean_binid7_pk_idx ON mangadapdb.cleanspaxelprop7 using BTREE(binid);
-- CREATE INDEX CONCURRENTLY clean_file7_pk_idx ON mangadapdb.cleanspaxelprop7 using BTREE(file_pk);
-- CREATE INDEX CONCURRENTLY clean_spaxel7_index_idx ON mangadapdb.cleanspaxelprop7 using BTREE(spaxel_index);
-- create index concurrently clean_spx7_x_idx on mangadapdb.cleanspaxelprop7 using btree(x);
-- create index concurrently clean_spx7_y_idx on mangadapdb.cleanspaxelprop7 using btree(y);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ha_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_hb_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_oiii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_sii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_oii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_oiia_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_nii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_stvel7_idx ON mangadapdb.cleanspaxelprop7 using BTREE(stellar_vel);
-- CREATE INDEX CONCURRENTLY clean_d40007_idx ON mangadapdb.cleanspaxelprop7 using BTREE(specindex_d4000);

-- CREATE INDEX CONCURRENTLY clean_stsig7_idx ON mangadapdb.cleanspaxelprop7 using BTREE(stellar_sigma);
-- CREATE INDEX CONCURRENTLY clean_emline7_gew_ha_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gew_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline7_gew_hb_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gew_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline7_gew_oiii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gew_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline7_gew_sii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gew_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline7_gew_oii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gew_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline7_gew_oiia_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gew_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline7_gew_nii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gew_nii_6585);

-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_oi_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_siia_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_sii_6732);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_ha_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_hb_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_oiii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_sii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_oii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_oiia_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_nii_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_oi_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline7_gflux_ivar_siia_idx ON mangadapdb.cleanspaxelprop7 using BTREE(emline_gflux_ivar_sii_6732);


-- # MPL-8

-- create table mangadapdb.cleanspaxelprop8 as select s.* from mangadapdb.spaxelprop8 as s
--     where (s.binid_binned_spectra != -1 and s.binid_stellar_continua != -1 and s.binid_spectral_indices != -1
--         and s.binid_em_line_moments != -1 and s.binid_em_line_models != -1);
-- alter table mangadapdb.cleanspaxelprop8 add constraint file_fk foreign key (file_pk) references mangadapdb.file(pk);

-- CREATE INDEX CONCURRENTLY clean_binid8_pk_idx ON mangadapdb.cleanspaxelprop8 using BTREE(binid);
-- CREATE INDEX CONCURRENTLY clean_file8_pk_idx ON mangadapdb.cleanspaxelprop8 using BTREE(file_pk);
-- CREATE INDEX CONCURRENTLY clean_spaxel8_index_idx ON mangadapdb.cleanspaxelprop8 using BTREE(spaxel_index);
-- create index concurrently clean_spx8_x_idx on mangadapdb.cleanspaxelprop8 using btree(x);
-- create index concurrently clean_spx8_y_idx on mangadapdb.cleanspaxelprop8 using btree(y);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ha_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_hb_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_oiii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_sii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_oii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_oiia_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_nii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_stvel8_idx ON mangadapdb.cleanspaxelprop8 using BTREE(stellar_vel);
-- CREATE INDEX CONCURRENTLY clean_d4000_8idx ON mangadapdb.cleanspaxelprop8 using BTREE(specindex_d4000);

-- CREATE INDEX CONCURRENTLY clean_stsig8_idx ON mangadapdb.cleanspaxelprop8 using BTREE(stellar_sigma);
-- CREATE INDEX CONCURRENTLY clean_emline8_gew_ha_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gew_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline8_gew_hb_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gew_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline8_gew_oiii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gew_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline8_gew_sii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gew_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline8_gew_oii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gew_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline8_gew_oiia_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gew_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline8_gew_nii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gew_nii_6585);

-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_oi_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_siia_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_sii_6732);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_ha_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_ha_6564);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_hb_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_hb_4862);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_oiii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_oiii_5008);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_sii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_sii_6718);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_oii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_oii_3727);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_oiia_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_oii_3729);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_nii_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_nii_6585);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_oi_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_oi_6302);
-- CREATE INDEX CONCURRENTLY clean_emline8_gflux_ivar_siia_idx ON mangadapdb.cleanspaxelprop8 using BTREE(emline_gflux_ivar_sii_6732);
