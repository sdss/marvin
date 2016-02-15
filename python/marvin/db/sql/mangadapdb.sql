
/*

mangaDapDB schema

Create Feb,2016 - B. Cherinka, J. Sanchez-Gallego, B. Andrews

*/


create schema mangadapdb;

set search_path to mangadapdb;

create table dap (pk serial primary key not null, cube_pk integer, pipeline_info_pk integer);

create table file (pk serial primary key not null, filename text, filepath text, num_ext integer, filetype_pk integer, dap_pk integer, structure_pk integer);

create table filetype (pk serial primary key not null, value text);

create table current_default (pk serial primary key not null, filename text, filepath text, file_pk integer);

create table hdu (pk serial primary key not null, extname_pk integer, exttype_pk integer, extno integer, file_pk integer);

create table hdu_to_header_value (pk serial primary key not null, hdu_pk integer, header_value_pk integer);

create table header_value (pk serial primary key not null, value text, index integer, comment text, header_keyword_pk integer, hdu_pk integer);

create table header_keyword (pk serial primary key not null, name text);

create table exttype (pk serial primary key not null, name text);

create table extname (pk serial primary key not null, name text);

create table hdu_to_extcol (pk serial primary key not null, hdu_pk integer, extcol_pk integer);

create table extcol (pk serial primary key not null, name text);

create table structure (pk serial primary key not null, binmode_pk integer, bintype_pk integer, template_kin_pk integer, template_pop_pk integer, executionplan_pk integer);

create table binid (pk serial primary key not null, index integer[][], structure_pk integer);

create table executionplan (pk serial primary key not null, id integer, comments text);

create table template (pk serial primary key not null, name text);

create table binmode (pk serial primary key not null, name text);

create table bintype (pk serial primary key not null, name text);

create table emline (pk serial primary key not null, value real[][], ivar real[][], mask integer[][], emline_parameter_pk integer, emline_type_pk integer, structure_pk integer);

create table emline_type (pk serial primary key not null, name text, rest_wavelength real, channel integer);

create table emline_parameter (pk serial primary key not null, name text, unit text);

create table stellar_kin (pk serial primary key not null, value real[][], ivar real[][], mask integer[][], stellar_kin_parameter_pk integer, stellar_kin_type_pk integer, structure_pk integer);

create table stellar_kin_type (pk serial primary key not null, name text, channel integer);

create table stellar_kin_parameter (pk serial primary key not null, name text, unit text);

create table stellar_pop (pk serial primary key not null, value real[][], ivar real[][], mask integer[][], stellar_pop_parameter_pk integer, stellar_pop_type_pk integer, structure_pk integer);

create table stellar_pop_type (pk serial primary key not null, name text, channel integer);

create table stellar_pop_parameter (pk serial primary key not null, name text, unit text);

create table specindex (pk serial primary key not null, value real[][], ivar real[][], mask integer[][], specindex_type_pk integer, structure_pk integer);

create table specindex_type (pk serial primary key not null, name text, channel integer, unit text);

/*
insert into mangadapdb.binmode values (0,'cube'),(1,'rss');
insert into mangadapdb.bintype values (0,'none'),(1,'ston');
*/

ALTER TABLE ONLY mangadapdb.dap
    ADD CONSTRAINT cube_fk
    FOREIGN KEY (cube_pk) REFERENCES mangadatadb.cube(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.dap
    ADD CONSTRAINT pipeline_info_fk
    FOREIGN KEY (pipeline_info_pk) REFERENCES mangadatadb.pipeline_info(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.file
    ADD CONSTRAINT dap_fk
    FOREIGN KEY (dap_pk) REFERENCES mangadapdb.dap(pk)
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

ALTER TABLE ONLY mangadapdb.binid
    ADD CONSTRAINT structure_fk
    FOREIGN KEY (structure_pk) REFERENCES mangadapdb.structure(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.file
    ADD CONSTRAINT structure_fk
    FOREIGN KEY (structure_pk) REFERENCES mangadapdb.structure(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.emline
    ADD CONSTRAINT emline_parameter_fk
    FOREIGN KEY (emline_parameter_pk) REFERENCES mangadapdb.emline_parameter(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.emline
    ADD CONSTRAINT emline_type_fk
    FOREIGN KEY (emline_type_pk) REFERENCES mangadapdb.emline_type(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.emline
    ADD CONSTRAINT structure_fk
    FOREIGN KEY (structure_pk) REFERENCES mangadapdb.structure(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.stellar_kin
    ADD CONSTRAINT stellar_kin_parameter_fk
    FOREIGN KEY (stellar_kin_parameter_pk) REFERENCES mangadapdb.stellar_kin_parameter(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.stellar_kin
    ADD CONSTRAINT stellar_kin_type_fk
    FOREIGN KEY (stellar_kin_type_pk) REFERENCES mangadapdb.stellar_kin_type(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.stellar_kin
    ADD CONSTRAINT structure_fk
    FOREIGN KEY (structure_pk) REFERENCES mangadapdb.structure(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.stellar_pop
    ADD CONSTRAINT stellar_pop_parameter_fk
    FOREIGN KEY (stellar_pop_parameter_pk) REFERENCES mangadapdb.stellar_pop_parameter(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.stellar_pop
    ADD CONSTRAINT stellar_pop_type_fk
    FOREIGN KEY (stellar_pop_type_pk) REFERENCES mangadapdb.stellar_pop_type(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.stellar_pop
    ADD CONSTRAINT structure_fk
    FOREIGN KEY (structure_pk) REFERENCES mangadapdb.structure(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.specindex
    ADD CONSTRAINT specindex_type_fk
    FOREIGN KEY (specindex_type_pk) REFERENCES mangadapdb.specindex_type(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;

ALTER TABLE ONLY mangadapdb.specindex
    ADD CONSTRAINT structure_fk
    FOREIGN KEY (structure_pk) REFERENCES mangadapdb.structure(pk)
    ON UPDATE CASCADE ON DELETE CASCADE;
