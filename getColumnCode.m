function [E]=getColumnCode(num)
    angka = num;
    E = '';
    while angka > 0
        sisa = mod((angka - 1), 26);
        E = strcat(char(65 + sisa),E);
        angka = floor((angka - sisa)/26);
    end
end